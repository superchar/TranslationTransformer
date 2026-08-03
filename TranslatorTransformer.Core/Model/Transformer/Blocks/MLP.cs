using TorchSharp.Modules;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.ModelConfiguration;

namespace TranslatorTransformer.Core.Model.Transformer.Blocks;

internal class MLP : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _mlp;

    public MLP() : base("MLP")
    {
        _mlp = nn.Sequential(
            nn.Linear(HiddenSize, MLPHidden),
            nn.ReLU(),
            nn.Linear(MLPHidden, HiddenSize));

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
        => _mlp.forward(input);
}