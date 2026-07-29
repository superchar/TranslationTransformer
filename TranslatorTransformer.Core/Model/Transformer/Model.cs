using TorchSharp;
using TorchSharp.Modules;

namespace TranslatorTransformer.Core.Model.Transformer;

public class TransformerInferenceModel : IInferenceModel
{
    public void Train(IEnumerable<string> documents)
    {
    }

    public IEnumerable<string> PerformInference(string prompt)
    {
        return [];
    }
}

internal class TransformerEncoder : torch.nn.Module<torch.Tensor, torch.Tensor>
{
    private readonly ModuleList<Block> _blocks;
    
    public TransformerEncoder() : base("Transformer encoder")
    {
        _blocks = torch.nn.ModuleList(Enumerable.Range(0, ModelConfiguration.NumBlocks)
            .Select(_ => new Block())
            .ToArray());
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        foreach (var block in _blocks)
        {
            input = block.forward(input, null, false);
        }
        
        return input;
    }
}

internal class Block : torch.nn.Module<torch.Tensor, torch.Tensor, bool, torch.Tensor>
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


    public override torch.Tensor forward(torch.Tensor tgt, torch.Tensor? src, bool isMasked)
    {
        tgt = _multiHeadSelfAttentionLayerNorm.forward(tgt + _multiHeadSelfAttention.forward(tgt, tgt, tgt, isMasked));

        if (src is not null)
        {
            tgt = _multiHeadCrossAttentionLayerNorm.forward(tgt +
                                                            _multiHeadCrossAttention.forward(tgt, src, src, isMasked));
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
    private readonly Linear _projection;
    private readonly ModuleList<AttentionHead> _heads;

    public MultiHeadedAttention() : base("Multi headed attention")
    {
        _projection = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HiddenSize);

        _heads = torch.nn.ModuleList(Enumerable.Range(0, ModelConfiguration.HeadSize)
            .Select(n => new AttentionHead(n))
            .ToArray());

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor querySrc, torch.Tensor keySrc, torch.Tensor valueSrc,
        bool isMasked)
    {
        var ouputs = _heads
            .Select(h => h.forward(querySrc, keySrc, valueSrc, isMasked))
            .ToList();

        var concatenatedOutputs = torch.cat(ouputs, 2);

        return _projection.forward(concatenatedOutputs);
    }
}

internal class AttentionHead : torch.nn.Module<torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor>
{
    private readonly Linear _key;
    private readonly Linear _query;
    private readonly Linear _value;

    private readonly double _scalingFactor;

    public AttentionHead(int number) : base($"AttentionHead({number})")
    {
        _key = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HeadSize);
        _query = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HeadSize);
        _value = torch.nn.Linear(ModelConfiguration.HiddenSize, ModelConfiguration.HeadSize);

        _scalingFactor = Math.Sqrt(ModelConfiguration.HeadSize);

        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor querySrc, torch.Tensor keySrc, torch.Tensor valueSrc,
        bool isMasked)
    {
        var query = _query.forward(querySrc);
        var key = _key.forward(keySrc);
        var value = _value.forward(valueSrc);

        var attentionScores = query.matmul(key.transpose(2, 1)) / _scalingFactor;
        if (isMasked)
        {
            var mask = torch.tril(torch.ones(query.shape[1], query.shape[1])).eq(0);
            mask = mask.to(DeviceManager.GetDevice());
            attentionScores = attentionScores.masked_fill(mask, float.NegativeInfinity);
        }

        return torch.softmax(attentionScores, 2).matmul(value);
    }
}