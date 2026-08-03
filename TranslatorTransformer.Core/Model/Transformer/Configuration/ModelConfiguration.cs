namespace TranslatorTransformer.Core.Model.Transformer.Configuration;

public static class ModelConfiguration
{
    public const int HiddenSize = 512;

    public const int NumHeads = 8;

    public const int NumBlocks = 8;

    public const int MLPHidden = 4 * HiddenSize;

    public const int MaxContextSize = 1024;

    public static int HeadSize => HiddenSize / NumHeads;
}