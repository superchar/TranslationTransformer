using TorchSharp.Modules;
using TranslatorTransformer.Core.Model.Transformer.Blocks;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.ModelConfiguration;

namespace TranslatorTransformer.Core.Model.Transformer;

internal class TransformerDecoder : nn.Module<Tensor, Tensor, Tensor, Tensor, Tensor>
{
    private readonly ModuleList<Block> _blocks;

    public TransformerDecoder() : base("Transformer decoder")
    {
        _blocks = nn.ModuleList(Enumerable.Range(0, NumBlocks)
            .Select(_ => new Block())
            .ToArray());

        RegisterComponents();
    }

    public override Tensor forward(Tensor input, Tensor encoderOutput, Tensor targetPaddingMask,
        Tensor sourcePaddingMask)
        => _blocks.Aggregate(input,
            (current, block) => block.forward(current, encoderOutput, targetPaddingMask, sourcePaddingMask, true));
}