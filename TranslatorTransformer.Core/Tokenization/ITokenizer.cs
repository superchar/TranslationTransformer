namespace TranslatorTransformer.Core.Tokenization;

public interface ITokenizer
{
    public static class SpecialTokens
    {
        public const string StartOfTheSequence = "<|START|>";

        public static string[] All = [StartOfTheSequence];
    }

    public const int VocabSize = 500;

    void Train(IEnumerable<string> documents, int vocabSize);

    List<int> Encode(string content);

    string Decode(int[] encoded);
}