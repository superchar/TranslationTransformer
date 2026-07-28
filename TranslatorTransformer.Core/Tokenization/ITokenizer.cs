namespace TranslatorTransformer.Core.Tokenization;

public interface ITokenizer
{
    void Train(IEnumerable<string> documents, int vocabSize);

    int[] Encode(string content);
    
    string Decode(int[] encoded);
}