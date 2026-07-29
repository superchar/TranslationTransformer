namespace TranslatorTransformer.Core.Tokenization;

public interface ITokenizer
{
    public const int VocabSize = 300;

    void Train(string content, int vocabSize);

    List<int> Encode(string content);

    string Decode(int[] encoded);
    
    public static class SpecialTokens
    {
        public const string StartOfTheSequence = "<|START|>";

        public static string[] All = [StartOfTheSequence];
    }
}