using TorchSharp;

namespace TranslatorTransformer.Core.Model.Transformer;

public static class DeviceManager
{
    public static string GetDevice() => torch.cuda_is_available() ? "cuda" : "cpu";
}