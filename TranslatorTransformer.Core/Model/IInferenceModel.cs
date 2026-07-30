namespace TranslatorTransformer.Core.Model;

public interface IInferenceModel
{
    void Train(List<(string Source, string Target)> documents);
    
    IEnumerable<int> PerformInference(string sourceText, string targetText);
}