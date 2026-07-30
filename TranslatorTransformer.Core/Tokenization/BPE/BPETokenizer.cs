using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TranslatorTransformer.Core.Tokenization.BPE;

public class BPETokenizer : ITokenizer
{
    private const string NonImplementedErrorMessage = $"The tokenizer was not trained. Call {nameof(Train)}() first.";
    private const string CachingFileName = "BPETokenizerCache.json";

    private static readonly string SpecialTokensPattern = 
        string.Join("|", ITokenizer.SpecialTokens.All.Select(Regex.Escape));
    
    private static readonly Regex WordSplittingRegex = new(
        $"{SpecialTokensPattern}|'s|'t|'re|'ve|'m|'ll|'d| ?\\p{{L}}+| ?\\p{{N}}+| ?(?:(?!(?:{SpecialTokensPattern}))[^\\s\\p{{L}}\\p{{N}}])+|\\s+(?!\\S)|\\s+\n",
        RegexOptions.Compiled);

    private Dictionary<Bytes, int>? _bytesToTokenMappingTable;
    private Dictionary<int, Bytes> _tokenToBytesMappingTable;

    public int VocabSize => _bytesToTokenMappingTable?.Count ?? 0;
    
    public void Train(string content, int vocabSize)
    {
        Console.WriteLine($"Training a tokenizer to be {vocabSize} words.");
        var cachedMergingTable = GetFromCache();
        if (cachedMergingTable != null)
        {
            Console.WriteLine("Used tokenizer cache");
            _bytesToTokenMappingTable = cachedMergingTable;
            _tokenToBytesMappingTable = _bytesToTokenMappingTable.ToDictionary(m => m.Value, m => m.Key);
            return;
        }
        
        _bytesToTokenMappingTable = new Dictionary<Bytes, int>();

        foreach (var basicByte in Enumerable.Range(byte.MinValue, byte.MaxValue + 1))
        {
            var bytes = new Bytes([(byte)basicByte]);
            _bytesToTokenMappingTable[bytes] = basicByte;
        }

        foreach (var specialToken in ITokenizer.SpecialTokens.All)
        {
            var bytes = GetWordBytes(specialToken);
            var mergedWord = bytes.Aggregate(new Bytes([]), (acc, b) => acc.Merge(b));
            _bytesToTokenMappingTable[mergedWord] = _bytesToTokenMappingTable.Count;
        }
        
        if (vocabSize <= byte.MaxValue)
        {
            _tokenToBytesMappingTable = _bytesToTokenMappingTable.ToDictionary(m => m.Value, m => m.Key);
            SetToCache(_bytesToTokenMappingTable);
            return;
        }

        var words = GetWords(content)
            .Where(w => w.Type == WordType.Regular)
            .GroupBy(w => w.Content)
            .Select(g => (Word: GetWordBytes(g.Key), Count: g.Count()))
            .ToList();

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

            Console.WriteLine($"{_bytesToTokenMappingTable.Count} tokens created.");
        }

        SetToCache(_bytesToTokenMappingTable);
        _tokenToBytesMappingTable = _bytesToTokenMappingTable.ToDictionary(m => m.Value, m => m.Key);
    }


    public List<int> Encode(string content)
    {
        ThrowIfNotTrained();

        var words = GetWordsBytes(content);
        for (var i = 0; i < words.Count; i++)
        {
            if (words[i].Type != WordType.SpecialToken)
            {
                continue;
            }
            
            var aggregatedBytes = words[i].Bytes.Aggregate(new Bytes([]), (acc, b) => acc.Merge(b));
            words[i] = words[i] with { Bytes = [aggregatedBytes] };
        }

        var regularWords = words
            .Where(w => w.Type == WordType.Regular)
            .ToList();
        
        while (true)
        {
            var leastFrequentPairMappingTable = GetPairFrequency(regularWords);

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

        return words.SelectMany(word => word.Bytes
                .Select(wordByte => _bytesToTokenMappingTable[wordByte]))
            .ToList();
    }

    public string Decode(List<int> encoded)
    {
        ThrowIfNotTrained();
        
        return  Encoding.UTF8.GetString(encoded
            .SelectMany(token => _tokenToBytesMappingTable[token].Data)
            .ToArray());
    }


    private static List<Word> GetWordsBytes(string content)
        => GetWords(content)
            .Select(w => w with { Bytes = GetWordBytes(w.Content) })
            .ToList();

    private static List<Bytes> GetWordBytes(string word)
        => Encoding.UTF8.GetBytes(word)
            .Select(b => new Bytes([b]))
            .ToList();

    private static List<Word> GetWords(string content)
    {
        return WordSplittingRegex
            .Matches(content)
            .Select(m =>
                new Word(ITokenizer.SpecialTokens.All.Contains(m.Value) ? WordType.SpecialToken : WordType.Regular,
                    m.Value))
            .ToList();
    }

    private static void MergeBytes(List<Word> words, Bytes first, Bytes second)
    {
        foreach (var word in words)
        {
            for (var i = 0; i < word.Bytes.Count - 1; i++)
            {
                if (!word.Bytes[i].Equals(first) ||
                    !word.Bytes[i + 1].Equals(second))
                {
                    continue;
                }

                word.Bytes[i] = first.Merge(second);
                word.Bytes.RemoveAt(i + 1);
                i--;
            }
        }
    }

    private static void MergeBytes(List<(List<Bytes> Word, int Count)> words, Bytes first, Bytes second)
    {
        foreach (var word in words)
        {
            for (var i = 0; i < word.Word.Count - 1; i++)
            {
                if (!word.Word[i].Equals(first) ||
                    !word.Word[i + 1].Equals(second))
                {
                    continue;
                }

                word.Word[i] = first.Merge(second);
                word.Word.RemoveAt(i + 1);
                i--;
            }
        }
    }

    private static Dictionary<(Bytes, Bytes), int> GetPairFrequency(List<Word> words)
    {
        var frequencyTable = new Dictionary<(Bytes, Bytes), int>();

        foreach (var word in words.SelectMany(word => word.Bytes.Zip(word.Bytes.Skip(1))))
        {
            frequencyTable[word] = frequencyTable.GetValueOrDefault(word, 0) + 1;
        }

        return frequencyTable;
    }

    private static Dictionary<(Bytes, Bytes), int> GetPairFrequency(List<(List<Bytes> Word, int Count)> words)
    {
        var frequencyTable = new Dictionary<(Bytes, Bytes), int>();

        foreach (var (word, count) in words)
        {
            for (var i = 0; i < word.Count - 1; i++)
            {
                var pair = (word[i], word[i + 1]);
                frequencyTable[pair] = frequencyTable.GetValueOrDefault(pair, 0) + count;
            }
        }

        return frequencyTable;
    }

    private static Dictionary<Bytes, int>? GetFromCache()
    {
        if (!File.Exists(CachingFileName))
        {
            return null;
        }
    
        var content = File.ReadAllText(CachingFileName);
        var intermediate = JsonSerializer.Deserialize<Dictionary<string, int>>(content);

        return intermediate?.ToDictionary(
            kvp => new Bytes(Convert.FromBase64String(kvp.Key)),
            kvp => kvp.Value
        );
    }

    private static void SetToCache(Dictionary<Bytes, int> bytesToTokenMappingTable)
    {
        var intermediate = bytesToTokenMappingTable.ToDictionary(
            kvp => Convert.ToBase64String(kvp.Key.Data.ToArray()),
            kvp => kvp.Value
        );
    
        var json = JsonSerializer.Serialize(intermediate);
        File.WriteAllText(CachingFileName, json);
    }

    private void ThrowIfNotTrained()
    {
        if (_bytesToTokenMappingTable is null)
        {
            throw new NotImplementedException(NonImplementedErrorMessage);
        }
    }
}