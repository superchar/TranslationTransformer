using TorchSharp;
using TranslatorTransformer.Core.Tokenization;
using TranslatorTransformer.Core.Utils;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.DeviceManager;

namespace TranslatorTransformer.Core.Model.Transformer;

public class TransformerInferenceModel : IInferenceModel
{
    private const int UTF8MaxCharLengthBytes = 4;
    private const float LearningRate = 0.0001f;

    private static readonly string CachingFileName = $"Cache{Path.DirectorySeparatorChar}TransformerWeightsCache.dat";

    private readonly int _paddingTokenId;
    private readonly int _encOfSequenceTokenId;
    private readonly TransformerEncoderDecoder _transformerEncoderDecoder = new();

    public TransformerInferenceModel(ITokenizer tokenizer)
    {
        Tokenizer = tokenizer;
        _transformerEncoderDecoder.to(GetDevice());
        _paddingTokenId = Tokenizer.Encode(ITokenizer.SpecialTokens.PaddingToken)[0];
        _encOfSequenceTokenId = Tokenizer.Encode(ITokenizer.SpecialTokens.EndOfTheSequence)[0];
    }

    public ITokenizer Tokenizer { get; }

    public void Train(List<(string Source, string Target)> documents, int numberOfIterations, bool useCache = true)
    {
        if (TryLoadModelFromCache(useCache))
        {
            return;
        }

        _transformerEncoderDecoder.train();

        using var optimizer = optim.AdamW(_transformerEncoderDecoder.parameters(), lr: LearningRate);

        foreach (var iteration in Enumerable.Range(0, numberOfIterations))
        {
            using var _ = NewDisposeScope();

            var (source, decoderInput, decoderTarget) = GetRandomBatch(documents);
            if (source is null || decoderInput is null || decoderTarget is null)
            {
                return;
            }

            var modelOutput = _transformerEncoderDecoder.forward(source, decoderInput, _paddingTokenId);

            var logits = modelOutput.view(-1, ITokenizer.VocabSize);
            var targets = decoderTarget.view(-1);

            var loss = nn.functional.cross_entropy(logits, targets, ignore_index: _paddingTokenId);
            
            optimizer.zero_grad();
            loss.backward();
            nn.utils.clip_grad_norm_(_transformerEncoderDecoder.parameters(), 1.0);
            optimizer.step();

            Console.WriteLine($"{iteration}) Loss value: {loss.item<float>():F4}");
        }

        AddModelToCache(useCache);
    }

    public IEnumerable<List<int>> PerformInference(string sourceText, string targetText)
    {
        using var _ = no_grad();
        using var __ = NewDisposeScope();
        _transformerEncoderDecoder.eval();
        if (string.IsNullOrWhiteSpace(targetText))
        {
            targetText = ITokenizer.SpecialTokens.StartOfTheSequence;
        }

        var sourceTextTokensTensor = ConvertTokensToBatch(Tokenizer.Encode(sourceText));
        var targetTextTokens = Tokenizer.Encode(targetText);
        
        var tokensBuffer = new List<int>();
        while (true)
        {
            var targetTextTokensTensor = ConvertTokensToBatch(targetTextTokens);
            var modelOutput =
                _transformerEncoderDecoder.forward(sourceTextTokensTensor, targetTextTokensTensor, _paddingTokenId);
            var logits = modelOutput[0, -1];
            var nextToken = (int)argmax(logits, -1).item<long>();
            if (nextToken == _encOfSequenceTokenId)
            {
                break;
            }

            tokensBuffer.Add(nextToken);
            if (tokensBuffer.Count >= UTF8MaxCharLengthBytes)
            {
                yield return tokensBuffer;
                tokensBuffer = [];
            }

            targetTextTokens.Add(nextToken);
        }

        if (tokensBuffer.Count > 0)
        {
            yield return tokensBuffer;
        }

        yield break;

        static Tensor ConvertTokensToBatch(List<int> tokens)
                => tensor(tokens)
                    .unsqueeze(0)
                    .to(GetDevice());
    }

    private (Tensor? Source, Tensor? DecoderInput, Tensor? DecoderTarget) GetRandomBatch(
        List<(string Source, string Target)> documents)
    {
        const int batchSize = 10;
        const int minSourceLength = 1;
        const int minTargetLength = 2;

        var batch = documents.TakeRandom(batchSize);

        var encodedBatch = batch
            .Select(b =>
            (
                SourceTokens: Tokenizer.Encode(b.Source).Select(t => (long)t).ToList(),
                TargetTokens: Tokenizer.Encode(b.Target).Select(t => (long)t).ToList()
            ))
            .Where(b => b.SourceTokens.Count >= minSourceLength && b.TargetTokens.Count >= minTargetLength).ToList();

        return encodedBatch.Count == 0 ? (null, null, null) : ReshapeEncodedBatchToTensors(encodedBatch);
    }

    private (Tensor? Source, Tensor? DecoderInput, Tensor? DecoderTarget) ReshapeEncodedBatchToTensors(
        List<(List<long> SourceTokens, List<long> TargetTokens)> encodedBatch)
    {
        var currentBatchSize = encodedBatch.Count;

        var maxSourceLength = encodedBatch.Max(b => b.SourceTokens.Count);
        var maxTargetLength = encodedBatch.Max(b => b.TargetTokens.Count) - 1;

        var sourceData = new long[currentBatchSize * maxSourceLength];
        var decoderInputData = new long[currentBatchSize * maxTargetLength];
        var decoderTargetData = new long[currentBatchSize * maxTargetLength];

        for (var batchIndex = 0; batchIndex < currentBatchSize; batchIndex++)
        {
            var sourceTokens = encodedBatch[batchIndex].SourceTokens;
            var targetTokens = encodedBatch[batchIndex].TargetTokens;

            for (var sourceIndex = 0; sourceIndex < maxSourceLength; sourceIndex++)
            {
                sourceData[batchIndex * maxSourceLength + sourceIndex] = sourceIndex < sourceTokens.Count
                    ? sourceTokens[sourceIndex]
                    : _paddingTokenId;
            }

            for (var targetIndex = 0; targetIndex < maxTargetLength; targetIndex++)
            {
                decoderInputData[batchIndex * maxTargetLength + targetIndex] = targetIndex < targetTokens.Count - 1
                    ? targetTokens[targetIndex]
                    : _paddingTokenId;
                decoderTargetData[batchIndex * maxTargetLength + targetIndex] = targetIndex + 1 < targetTokens.Count
                    ? targetTokens[targetIndex + 1]
                    : _paddingTokenId;
            }
        }

        var sourceTensor = tensor(sourceData, new long[] { currentBatchSize, maxSourceLength }).to(GetDevice());
        var decoderInputTensor =
            tensor(decoderInputData, new long[] { currentBatchSize, maxTargetLength }).to(GetDevice());
        var decoderTargetTensor =
            tensor(decoderTargetData, new long[] { currentBatchSize, maxTargetLength }).to(GetDevice());

        return (sourceTensor, decoderInputTensor, decoderTargetTensor);
    }

    private bool TryLoadModelFromCache(bool useCache)
    {
        if (!useCache || !File.Exists(CachingFileName))
        {
            return false;
        }

        Console.WriteLine("Used model weights cache.");
        _transformerEncoderDecoder.load(CachingFileName);
        return true;
    }

    private void AddModelToCache(bool useCache)
    {
        if (!useCache)
        {
            return;
        }

        _transformerEncoderDecoder.save(CachingFileName);
        Console.WriteLine("Model weights saved to internal cache successfully.");
    }
}