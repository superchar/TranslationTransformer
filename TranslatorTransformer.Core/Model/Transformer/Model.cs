using TorchSharp;
using TranslatorTransformer.Core.Model.Transformer.Configuration;
using TranslatorTransformer.Core.Tokenization;

namespace TranslatorTransformer.Core.Model.Transformer;

public class TransformerInferenceModel : IInferenceModel
{
    private static readonly string CachingFileName = $"Cache{Path.DirectorySeparatorChar}TransformerWeightsCache.dat";
    
    private readonly ITokenizer _tokenizer;
    private readonly TransformerEncoderDecoder _transformerEncoderDecoder = new();

    public TransformerInferenceModel(ITokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _transformerEncoderDecoder.to(DeviceManager.GetDevice());
    }

    public ITokenizer Tokenizer => _tokenizer;
    
    public void Train(List<(string Source, string Target)> documents, int numberOfIterations, bool useCache = true)
    {
        if (File.Exists(CachingFileName) && useCache)
        {
            Console.WriteLine("Used model weights cache.");
            _transformerEncoderDecoder.load(CachingFileName);
            return; 
        }
        
        const int batchSize = 10;
        var paddingTokenId = _tokenizer.Encode(ITokenizer.SpecialTokens.PaddingToken)[0];
        _transformerEncoderDecoder.train();

        const double learningRate = 1e-4;
        using var optimizer = torch.optim.Adam(_transformerEncoderDecoder.parameters(), lr: learningRate);

        foreach (var iteration in Enumerable.Range(0, numberOfIterations))
        {
            var random = new Random();
             var batch = documents.Count <= batchSize ? documents : Enumerable.Range(0, batchSize)
                .Select(_ => documents[random.Next(documents.Count)])
                .ToList();
            using var scope = torch.NewDisposeScope();

            var encodedBatch = batch.Select(b => new
            {
                SourceTokens = _tokenizer.Encode(b.Source).Select(t => (long)t).ToList(),
                TargetTokens = _tokenizer.Encode(b.Target).Select(t => (long)t).ToList()
            }).Where(b => b.SourceTokens.Count > 0 && b.TargetTokens.Count >= 2).ToList();

            if (encodedBatch.Count == 0)
            {
                continue;
            }

            var currentBatchSize = encodedBatch.Count;

            var maxSourceLen = encodedBatch.Max(b => b.SourceTokens.Count);
            var maxTargetLen = encodedBatch.Max(b => b.TargetTokens.Count) - 1;

            var sourceData = new long[currentBatchSize * maxSourceLen];
            var decoderInputData = new long[currentBatchSize * maxTargetLen];
            var decoderTargetData = new long[currentBatchSize * maxTargetLen];

            for (var b = 0; b < currentBatchSize; b++)
            {
                var src = encodedBatch[b].SourceTokens;
                var tgt = encodedBatch[b].TargetTokens;

                for (var s = 0; s < maxSourceLen; s++)
                {
                    sourceData[b * maxSourceLen + s] = s < src.Count ? src[s] : paddingTokenId;
                }

                for (var t = 0; t < maxTargetLen; t++)
                {
                    decoderInputData[b * maxTargetLen + t] = t < tgt.Count - 1 ? tgt[t] : paddingTokenId;
                    decoderTargetData[b * maxTargetLen + t] = t + 1 < tgt.Count ? tgt[t + 1] : paddingTokenId;
                }
            }

            var sourceTensor = torch.tensor(sourceData, new long[] { currentBatchSize, maxSourceLen })
                .to(DeviceManager.GetDevice());
            var decoderInputTensor = torch.tensor(decoderInputData, new long[] { currentBatchSize, maxTargetLen })
                .to(DeviceManager.GetDevice());
            var decoderTargetTensor = torch.tensor(decoderTargetData, new long[] { currentBatchSize, maxTargetLen })
                .to(DeviceManager.GetDevice());

            var modelOutput = _transformerEncoderDecoder.forward(sourceTensor, decoderInputTensor, paddingTokenId);

            var logits = modelOutput.view(-1, ITokenizer.VocabSize);
            var targets = decoderTargetTensor.view(-1);

            var loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index: paddingTokenId);
            

            optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(_transformerEncoderDecoder.parameters(), 1.0);
            optimizer.step();

            Console.WriteLine($"{iteration}) Loss value: {loss.item<float>():F4}");
        }

        if (useCache)
        {
            _transformerEncoderDecoder.save(CachingFileName);
            Console.WriteLine("Model weights saved to internal cache successfully.");
        }
    }

    public IEnumerable<int> PerformInference(string sourceText, string targetText)
    {
        using var gradScope = torch.no_grad();
        _transformerEncoderDecoder.eval();
        if (string.IsNullOrWhiteSpace(targetText))
        {
            targetText = ITokenizer.SpecialTokens.StartOfTheSequence;
        }

        var sourceTextTokensTensor = torch.tensor(_tokenizer.Encode(sourceText))
            .unsqueeze(0)
            .to(DeviceManager.GetDevice());
        var targetTextTokens = _tokenizer.Encode(targetText);

        var endOfSequenceToken = _tokenizer.Encode(ITokenizer.SpecialTokens.EndOfTheSequence)[0];
        var paddingTokenId = _tokenizer.Encode(ITokenizer.SpecialTokens.PaddingToken)[0];

        while (true)
        {
            using var scope = torch.NewDisposeScope();
            var targetTextTokensTensor = torch.tensor(targetTextTokens)
                .unsqueeze(0)
                .to(DeviceManager.GetDevice());
            var modelOutput =
                _transformerEncoderDecoder.forward(sourceTextTokensTensor, targetTextTokensTensor, paddingTokenId);
            var logits = modelOutput[0, -1];
            var nextToken = (int)torch.argmax(logits, -1).item<long>();
            if (nextToken == endOfSequenceToken)
            {
                break;
            }

            yield return nextToken;
            targetTextTokens.Add(nextToken);
        }
    }
}