using TorchSharp.Modules;
using TranslatorTransformer.Core.Model.Transformer.Configuration;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.ModelConfiguration;

namespace TranslatorTransformer.Core.Model.Transformer.Blocks;

internal class MultiHeadedAttention : nn.Module<Tensor, Tensor, Tensor, Tensor, bool, Tensor>
{
    private readonly double _scalingFactor;

    private readonly Linear _queryLinear;
    private readonly Linear _keyLinear;
    private readonly Linear _valueLinear;
    private readonly Linear _outProjection;

    public MultiHeadedAttention() : base("Multi headed attention")
    {
        _scalingFactor = Math.Sqrt(HeadSize);

        _queryLinear = nn.Linear(HiddenSize, HiddenSize);
        _keyLinear = nn.Linear(HiddenSize, HiddenSize);
        _valueLinear = nn.Linear(HiddenSize, HiddenSize);

        _outProjection = nn.Linear(HiddenSize, HiddenSize);

        var initialMask = tril(ones(MaxContextSize, MaxContextSize,
            device: DeviceManager.GetDevice())).eq(0);
        register_buffer("attention_mask", initialMask);

        RegisterComponents();
    }

    public override Tensor forward(Tensor querySrc, Tensor keySrc, Tensor valueSrc, Tensor? paddingMask,
        bool selfAttentionMaskEnabled)
    {
        var batchSize = querySrc.shape[0];
        var query = _queryLinear.forward(querySrc);
        var key = _keyLinear.forward(keySrc);
        var value = _valueLinear.forward(valueSrc);

        query = query.view(batchSize, querySrc.shape[1], NumHeads, HeadSize).transpose(1, 2);
        key = key.view(batchSize, keySrc.shape[1], NumHeads, HeadSize).transpose(1, 2);
        value = value.view(batchSize, valueSrc.shape[1], NumHeads, HeadSize).transpose(1, 2);

        var scores = query.matmul(key.transpose(-2, -1)) / _scalingFactor;

        if (paddingMask is not null)
        {
            scores = scores.masked_fill(paddingMask, float.NegativeInfinity);
        }

        if (selfAttentionMaskEnabled)
        {
            var attentionMask = get_buffer("attention_mask");
            var mask = attentionMask.narrow(0, 0, valueSrc.shape[1]).narrow(1, 0, valueSrc.shape[1]);
            scores = scores.masked_fill(mask, float.NegativeInfinity);
        }

        var attention = softmax(scores, -1);
        var context = attention.matmul(value);

        context = context.transpose(1, 2).contiguous().view(batchSize, querySrc.shape[1], HiddenSize);

        return _outProjection.forward(context);
    }
}