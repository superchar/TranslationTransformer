namespace TranslatorTransformer.Core.Model.Transformer.Configuration;

using TorchSharp;

public static class DeviceManager
{
    public static string GetDevice() => torch.cuda_is_available() ? "cuda" : "cpu";
}