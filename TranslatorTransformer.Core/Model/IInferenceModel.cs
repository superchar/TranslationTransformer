namespace TranslatorTransformer.Core.Model;

public interface IInferenceModel
{
    void Train(IEnumerable<string> documents);
    
    IEnumerable<string> PerformInference(string prompt);
}