using TorchSharp.Modules;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.ModelConfiguration;

namespace TranslatorTransformer.Core.Model.Transformer.Blocks;

internal class Block : nn.Module<Tensor, Tensor, Tensor, Tensor, bool, Tensor>
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
        _mlpLayerNorm = nn.LayerNorm(HiddenSize);

        _multiHeadSelfAttention = new MultiHeadedAttention();
        _multiHeadSelfAttentionLayerNorm = nn.LayerNorm(HiddenSize);

        _multiHeadCrossAttention = new MultiHeadedAttention();
        _multiHeadCrossAttentionLayerNorm = nn.LayerNorm(HiddenSize);

        RegisterComponents();
    }

    public override Tensor forward(Tensor target, Tensor? source, Tensor? selfAttentionPaddingMask,
        Tensor? crossAttentionPaddingMask, bool selfAttentionMaskEnabled)
    {
        var normalizedTarget = _multiHeadSelfAttentionLayerNorm.forward(target);
        target = target + _multiHeadSelfAttention.forward(normalizedTarget, normalizedTarget, normalizedTarget,
            selfAttentionPaddingMask, selfAttentionMaskEnabled);

        if (source is not null)
        {
            normalizedTarget = _multiHeadCrossAttentionLayerNorm.forward(target);
            target = target + _multiHeadCrossAttention.forward(normalizedTarget, source, source,
                crossAttentionPaddingMask, false);
        }

        normalizedTarget = _mlpLayerNorm.forward(target);
        return target + _mlp.forward(normalizedTarget);
    }
}