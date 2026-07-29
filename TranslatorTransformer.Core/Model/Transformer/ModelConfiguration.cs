namespace TranslatorTransformer.Core.Model.Transformer;

public static class ModelConfiguration
{
    public const int HiddenSize = 512;

    public const int NumHeads = 8;

    public const int NumBlocks = 8;

    public const int MLPScalingFactor = 4;

    public static int HeadSize => HiddenSize / NumHeads;
}