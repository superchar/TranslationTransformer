namespace TranslatorTransformer.Core.Utils;

public static class EnumerableExtensions
{
    public static IEnumerable<T> TakeRandom<T>(this IEnumerable<T> source, int count)
    {
        var random = new Random();

        return
            source
                .OrderBy(_ => random.Next())
                .Take(count);
    }
}