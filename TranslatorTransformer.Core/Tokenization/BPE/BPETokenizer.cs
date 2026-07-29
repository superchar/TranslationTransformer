using System.Text;
using System.Text.RegularExpressions;

namespace TranslatorTransformer.Core.Tokenization.BPE;

public partial class BPETokenizer : ITokenizer
{
    private const string NonImplementedErrorMessage = $"The tokenizer was not trained. Call {nameof(Train)}() first.";

    private Dictionary<Bytes, int>? _bytesToTokenMappingTable;
    private Dictionary<int, Bytes> _tokenToBytesMappingTable;

    public void Train(IEnumerable<string> documents, int vocabSize)
    {
        _bytesToTokenMappingTable = new Dictionary<Bytes, int>();

        foreach (var basicByte in Enumerable.Range(byte.MinValue, byte.MaxValue + 1))
        {
            var bytes = new Bytes([(byte)basicByte]);
            _bytesToTokenMappingTable[bytes] = basicByte;
        }

        foreach (var specialToken in ITokenizer.SpecialTokens.All)
        {
            _bytesToTokenMappingTable[new Bytes(Encoding.UTF8.GetBytes(specialToken))] = _bytesToTokenMappingTable.Count;
        }

        if (vocabSize <= byte.MaxValue)
        {
            return;
        }

        foreach (var document in documents)
        {
            if (_bytesToTokenMappingTable.Count == vocabSize)
            {
                break;
            }

            var words = GetWords(document);

            while (true)
            {
                var mergingByCountTable = GetPairFrequency(words);

                if (mergingByCountTable.Count == 0 || _bytesToTokenMappingTable.Count == vocabSize)
                {
                    break;
                }

                var mostFrequentPair = mergingByCountTable.MaxBy(pair => pair.Value);
                var firstMostFrequentByte = mostFrequentPair.Key.Item1;
                var secondMostFrequentByte = mostFrequentPair.Key.Item2;
                var mergedPair = firstMostFrequentByte.Merge(secondMostFrequentByte);
                _bytesToTokenMappingTable[mergedPair] = _bytesToTokenMappingTable.Count;
                
                MergeBytes(words, firstMostFrequentByte, secondMostFrequentByte);
            }
        }

        _tokenToBytesMappingTable = _bytesToTokenMappingTable.ToDictionary(m => m.Value, m => m.Key);
    }

    public List<int> Encode(string content)
    {
        ThrowIfNotTrained();

        var words = GetWords(content);

        while (true)
        {
            var leastFrequentPairMappingTable = GetPairFrequency(words);
          
            if (leastFrequentPairMappingTable.Count == 0)
            {
                break;
            }

            var minRank = int.MaxValue;
            Bytes? firstLeastFrequentByte = null;
            Bytes? secondLeastFrequentByte = null;
            foreach (var pair in leastFrequentPairMappingTable.OrderBy(pair => pair.Value))
            {
                var mergedPair = pair.Key.Item1.Merge(pair.Key.Item2);
                if (!_bytesToTokenMappingTable.TryGetValue(mergedPair, out var rank) || rank >= minRank)
                {
                    continue;
                }
                
                minRank = rank;
                firstLeastFrequentByte = pair.Key.Item1;
                secondLeastFrequentByte = pair.Key.Item2;
            }

            if (minRank == int.MaxValue)
            {
                break;
            }

            MergeBytes(words, firstLeastFrequentByte, secondLeastFrequentByte);
            
        }

        return words.SelectMany(word => word
                .Select(wordByte => _bytesToTokenMappingTable[wordByte]))
            .ToList();
    }

    public string Decode(int[] encoded)
        => Encoding.UTF8.GetString(encoded
            .SelectMany(token => _tokenToBytesMappingTable[token].Data)
            .ToArray());
    

    [GeneratedRegex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+\n",
        RegexOptions.Compiled)]
    private static partial Regex WordSplittingRegex();

    private static List<List<Bytes>> GetWords(string content)
    {
        var wordSplittingRegex = WordSplittingRegex();

        return wordSplittingRegex
            .Matches(content)
            .Select(word => Encoding.UTF8.GetBytes(word.Value)
                .Select(b => new Bytes([b]))
                .ToList())
            .ToList();
    }

    private static void MergeBytes(List<List<Bytes>> words, Bytes first, Bytes second)
    {
        foreach (var word in words)
        {
            for (var i = 0; i < word.Count - 1; i++)
            {
                if (!word[i].Equals(first) ||
                    !word[i + 1].Equals(second))
                {
                    continue;
                }

                word[i] = first.Merge(second);
                word.RemoveAt(i + 1);
                i--;
            }
        }
    }

    private static Dictionary<(Bytes, Bytes), int> GetPairFrequency(List<List<Bytes>>  words)
    {
        var frequencyTable = new Dictionary<(Bytes, Bytes), int>();
        
        foreach (var word in words.SelectMany(word => word.Zip(word.Skip(1))))
        {
            frequencyTable[word] = frequencyTable.GetValueOrDefault(word, 0) + 1;
        }

        return frequencyTable;
    }
    
    private void ThrowIfNotTrained()
    {
        if (_bytesToTokenMappingTable is null)
        {
            throw new NotImplementedException(NonImplementedErrorMessage);
        }
    }
}