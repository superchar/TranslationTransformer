namespace TranslatorTransformer.Core.Tokenization.BPE;

public record Bytes(byte[] Data)
{
    public Bytes Merge(Bytes other) => new (Data.Concat(other.Data).ToArray());

    public virtual bool Equals(Bytes? other) =>  other is not null && Data.SequenceEqual(other.Data);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.AddBytes(Data);
        return hash.ToHashCode();
    }
    
    public  override string ToString() => $"[{string.Join(",", Data)}]";
}