namespace TranslatorTransformer.Core.Tokenization.BPE;

public record Word(WordType Type, string Content, List<Bytes> Bytes = null);

public enum WordType
{
    Regular, 
    SpecialToken
}