using TorchSharp.Modules;
using TranslatorTransformer.Core.Model.Transformer.Blocks;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.ModelConfiguration;

namespace TranslatorTransformer.Core.Model.Transformer;

internal class TransformerEncoder : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly ModuleList<Block> _blocks;

    public TransformerEncoder() : base("Transformer encoder")
    {
        _blocks = nn.ModuleList(Enumerable.Range(0, NumBlocks)
            .Select(_ => new Block())
            .ToArray());

        RegisterComponents();
    }

    public override Tensor forward(Tensor input, Tensor sourcePaddingMask)
        => _blocks.Aggregate(input, (current, block) => block.forward(current, null, sourcePaddingMask, null, false));
}