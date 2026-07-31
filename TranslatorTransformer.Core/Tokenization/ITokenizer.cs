namespace TranslatorTransformer.Core.Tokenization;

public interface ITokenizer
{
    public const int VocabSize = 20_000;

    void Train(string content, int vocabSize, bool useCache = true);

    List<int> Encode(string content);

    string Decode(List<int> encoded);
    
    public static class SpecialTokens
    {
        public const string StartOfTheSequence = "<START>";
        
        public const string EndOfTheSequence = "<END>";
        
        public const string PaddingToken = "<PADDING>";

        public static readonly string[] All = [StartOfTheSequence, EndOfTheSequence, PaddingToken];
    }
}