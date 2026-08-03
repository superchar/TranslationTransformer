namespace TranslatorTransformer.Core.Model;

public interface IInferenceModel
{
    void Train(List<(string Source, string Target)> documents, int numberOfIterations,  bool useCache = true);
    
    IEnumerable<List<int>> PerformInference(string sourceText, string targetText);
}