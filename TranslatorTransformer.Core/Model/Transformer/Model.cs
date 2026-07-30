using TorchSharp;
using TorchSharp.Modules;
using TranslatorTransformer.Core.Tokenization;

namespace TranslatorTransformer.Core.Model.Transformer;

public class TransformerInferenceModel : IInferenceModel
{
    private readonly ITokenizer _tokenizer;
    private readonly EncoderDecoderTransformer _encoderDecoderTransformer = new();

    public TransformerInferenceModel(ITokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _encoderDecoderTransformer.to(DeviceManager.GetDevice());
    }

    public void Train(List<(string Source, string Target)> documents)
    {
        const int numberOfIterations = 10_000;
        const int batchSize = 10;
        var paddingTokenId = _tokenizer.Encode(ITokenizer.SpecialTokens.PaddingToken)[0];
        _encoderDecoderTransformer.train();

        const double learningRate = 1e-4;
        using var optimizer = torch.optim.Adam(_encoderDecoderTransformer.parameters(), lr: learningRate);

        foreach (var iteration in Enumerable.Range(0, numberOfIterations))
        {
            var random = new Random();
            var offset = random.Next(0, documents.Count - batchSize);
            var batch = documents.Skip(offset).Take(batchSize);
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

            var modelOutput = _encoderDecoderTransformer.forward(sourceTensor, decoderInputTensor, paddingTokenId);

            var logits = modelOutput.view(-1, ITokenizer.VocabSize);
            var targets = decoderTargetTensor.view(-1);

            var loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index: paddingTokenId);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            Console.WriteLine($"{iteration}) Loss value: {loss.item<float>():F4}");
        }
    }

    public IEnumerable<int> PerformInference(string sourceText, string targetText)
    {
        using var gradScope = torch.no_grad();
        _encoderDecoderTransformer.eval();
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
            var modelOutput = _encoderDecoderTransformer.forward(sourceTextTokensTensor, targetTextTokensTensor, paddingTokenId);
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

internal class EncoderDecoderTransformer : torch.nn.Module<torch.Tensor, torch.Tensor, long, torch.Tensor>
{
    private readonly Linear _linear = torch.nn.Linear(ModelConfiguration.HiddenSize, ITokenizer.VocabSize);

    private readonly TransformerEncoder _encoder = new();

    private readonly Embedding _encoderPositionalEmbedding =
        torch.nn.Embedding(ModelConfiguration.MaxContextSize, ModelConfiguration.HiddenSize);

    private readonly Embedding _encoderTokenEmbedding =
        torch.nn.Embedding(ITokenizer.VocabSize, ModelConfiguration.HiddenSize);

    private readonly TransformerDecoder _decoder = new();

    private readonly Embedding _decoderPositionalEmbedding =
        torch.nn.Embedding(ModelConfiguration.MaxContextSize, ModelConfiguration.HiddenSize);

    private readonly Embedding _decoderTokenEmbedding =
        torch.nn.Embedding(ITokenizer.VocabSize, ModelConfiguration.HiddenSize);

    public EncoderDecoderTransformer() : base("Encoder decoder transformer")
    {
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor encoderInput, torch.Tensor decoderInput, long paddingTokenId)
    {
        var srcPaddingMask = encoderInput.eq(paddingTokenId).unsqueeze(1).unsqueeze(2);
        var tgtPaddingMask = decoderInput.eq(paddingTokenId).unsqueeze(1).unsqueeze(2);

        var encoderPositions = torch.arange(encoderInput.shape[1], device: encoderInput.device, dtype: torch.ScalarType.Int64).unsqueeze(0);
        var encoderEmbedding = _encoderPositionalEmbedding.forward(encoderPositions) + _encoderTokenEmbedding.forward(encoderInput);
        var encoderOutput = _encoder.forward(encoderEmbedding, srcPaddingMask);

        var decoderPositions = torch.arange(decoderInput.shape[1], device: decoderInput.device, dtype: torch.ScalarType.Int64).unsqueeze(0);
        var decoderEmbedding = _decoderPositionalEmbedding.forward(decoderPositions) + _decoderTokenEmbedding.forward(decoderInput);
        var decoderOutput = _decoder.forward(decoderEmbedding, encoderOutput, tgtPaddingMask, srcPaddingMask);

        return _linear.forward(decoderOutput);
    }
}

internal class TransformerDecoder : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor>
{
    private readonly ModuleList<Block> _blocks;

    public TransformerDecoder() : base("Transformer decoder")
    {
        _blocks = torch.nn.ModuleList(Enumerable.Range(0, ModelConfiguration.NumBlocks)
            .Select(_ => new Block())
            .ToArray());
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, torch.Tensor encoderOutput, torch.Tensor tgtPaddingMask, torch.Tensor srcPaddingMask)
    {
        return _blocks.Aggregate(input, (current, block) => block.forward(current, encoderOutput, tgtPaddingMask, srcPaddingMask, true));
    }
}

internal class TransformerEncoder : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor>
{
    private readonly ModuleList<Block> _blocks;

    public TransformerEncoder() : base("Transformer encoder")
    {
        _blocks = torch.nn.ModuleList(Enumerable.Range(0, ModelConfiguration.NumBlocks)
            .Select(_ => new Block())
            .ToArray());
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input, torch.Tensor srcPaddingMask)
    {
        foreach (var block in _blocks)
        {
            input = block.forward(input, null, srcPaddingMask, null, false);
        }
        return input;
    }
}

internal class Block : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor>
{
    private readonly MLP _mlp;
    private readonly LayerNorm _mlpLayerNorm;

    private readonly LayerNorm _multiHeadSelfAttentionLayerNorm;
    private readonly MultiHeadedAttention _multiHeadSelfAttention;

    private readonly LayerNorm _multiHeadCrossAttentionLayerNorm;
    private readonly MultiHeadedAttention _multiHeadCrossAttention;


    public Block() : base("Block")
    {
        _mlp = new MLP();
        _mlpLayerNorm = torch.nn.LayerNorm(ModelConfiguration.HiddenSize);

        _multiHeadSelfAttention = new MultiHeadedAttention();
        _multiHeadSelfAttentionLayerNorm = torch.nn.LayerNorm(ModelConfiguration.HiddenSize);

        _multiHeadCrossAttention = new MultiHeadedAttention();
        _multiHeadCrossAttentionLayerNorm = torch.nn.LayerNorm(ModelConfiguration.HiddenSize);

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor tgt, torch.Tensor? src, torch.Tensor? selfAttentionMask, torch.Tensor? crossAttentionMask, bool isCausalMasked)
    {
        tgt = _multiHeadSelfAttentionLayerNorm.forward(tgt + _multiHeadSelfAttention.forward(tgt, tgt, tgt, selfAttentionMask, isCausalMasked));

        if (src is not null)
        {
            tgt = _multiHeadCrossAttentionLayerNorm.forward(tgt + _multiHeadCrossAttention.forward(tgt, src, src, crossAttentionMask, false));
        }

        return _mlpLayerNorm.forward(tgt + _mlp.forward(tgt));
    }
}

internal class MLP : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    private readonly Sequential _mlp;

    public MLP() : base("MLP")
    {
        const int scalingFactor = ModelConfiguration.HiddenSize * ModelConfiguration.MLPScalingFactor;
        _mlp = torch.nn.Sequential([
            torch.nn.Linear(ModelConfiguration.HiddenSize, scalingFactor),
            torch.nn.ReLU(),
            torch.nn.Linear(scalingFactor, ModelConfiguration.HiddenSize)
        ]);

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input)
        => _mlp.forward(input);
}

internal class MultiHeadedAttention : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor>
{
    private readonly int _numHeads;
    private readonly int _headSize;
    private readonly double _scalingFactor;

    private readonly Linear _qProj;
    private readonly Linear _kProj;
    private readonly Linear _vProj;
    private readonly Linear _outProj;

    public MultiHeadedAttention() : base("Multi headed attention")
    {
        _numHeads = ModelConfiguration.NumHeads;
        // Ensure HiddenSize is evenly divisible by NumHeads
        _headSize = ModelConfiguration.HiddenSize / ModelConfiguration.NumHeads;
        _scalingFactor = Math.Sqrt(_headSize);

        // Project the entire hidden dimension at once
        _qProj = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HiddenSize);
        _kProj = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HiddenSize);
        _vProj = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HiddenSize);

        _outProj = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HiddenSize);

        // Initialize locally just to register it. Do not store it in a C# class field.
        var initialMask = torch.tril(torch.ones(ModelConfiguration.MaxContextSize, ModelConfiguration.MaxContextSize,
            device: DeviceManager.GetDevice())).eq(0);
        register_buffer("causal_mask", initialMask);

        RegisterComponents();
    }

    public torch.Tensor forward(torch.Tensor querySrc, torch.Tensor keySrc, torch.Tensor valueSrc,
        torch.Tensor? paddingMask, bool isMasked)
    {
        var batchSize = querySrc.shape[0];
        var seqLenQ = querySrc.shape[1];
        var seqLenKV = keySrc.shape[1];

        var q = _qProj.forward(querySrc);
        var k = _kProj.forward(keySrc);
        var v = _vProj.forward(valueSrc);

        q = q.view(batchSize, seqLenQ, _numHeads, _headSize).transpose(1, 2);
        k = k.view(batchSize, seqLenKV, _numHeads, _headSize).transpose(1, 2);
        v = v.view(batchSize, seqLenKV, _numHeads, _headSize).transpose(1, 2);

        var kTransposed = k.transpose(-2, -1);
        var scores = q.matmul(kTransposed) / _scalingFactor;

        if (paddingMask is not null)
        {
            scores = scores.masked_fill(paddingMask, float.NegativeInfinity);
        }

        if (isMasked)
        {
            var activeMask = get_buffer("causal_mask");
            var mask = activeMask.narrow(0, 0, seqLenQ).narrow(1, 0, seqLenKV);
            scores = scores.masked_fill(mask, float.NegativeInfinity);
        }

        var attention = torch.softmax(scores, -1);
        var context = attention.matmul(v);

        context = context.transpose(1, 2).contiguous().view(batchSize, seqLenQ, ModelConfiguration.HiddenSize);

        return _outProj.forward(context);
    }

    // Retain generic compatibility layer for Module generic definitions
    public override torch.Tensor forward(torch.Tensor querySrc, torch.Tensor keySrc, torch.Tensor valueSrc,
        bool isMasked)
        => forward(querySrc, keySrc, valueSrc, null, isMasked);
}