namespace TranslatorTransformer.Core.Model;

public interface IInferenceModel
{
    void Train(IEnumerable<(string Source, string Target)> documents);
    
    IEnumerable<string> PerformInference(string sourceText, string targetText);
}