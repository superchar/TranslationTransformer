using System.Text;
using TranslatorTransformer.Core.Model.Transformer;
using TranslatorTransformer.Core.Tokenization;
using TranslatorTransformer.Core.Tokenization.BPE;

Console.OutputEncoding = Encoding.UTF8;

const int UTF8MaxCharLengthBytes = 4;

var model = GetModel();

while (true)
{
    
    Console.Write("Enter the text: ");
    var sourceText = Console.ReadLine();
    var tokensBuffer = new List<int>();

    foreach (var token in model.PerformInference(sourceText, string.Empty))
    {
        tokensBuffer.Add(token);
        if (tokensBuffer.Count >= UTF8MaxCharLengthBytes)
        {
            tokensBuffer.Clear();
        }
    }

    if (tokensBuffer.Count > 0)
    {
        Console.Write(model.Tokenizer.Decode(tokensBuffer));
    }
    
    Console.WriteLine();
}

TransformerInferenceModel GetModel()
{
    const int numberOfIterations = 100;
    const char lineSeparator = '\n';
    const string translationFolder = "Translations";

    var engText = File.ReadAllText($"{translationFolder}{Path.DirectorySeparatorChar}EN.txt");
    var rusText = File.ReadAllText($"{translationFolder}{Path.DirectorySeparatorChar}/RU.txt");
    
    var tokenizer = new BPETokenizer();
    tokenizer.Train(engText + rusText, ITokenizer.VocabSize);
    var model = new TransformerInferenceModel(tokenizer);

    var engLines = engText.Split(lineSeparator);
    var rusLines = rusText.Split(lineSeparator);

    model.Train(
        engLines.Zip(rusLines.Select(l =>
        {
            var stringBuilder = new StringBuilder();
            stringBuilder.Append(ITokenizer.SpecialTokens.StartOfTheSequence);
            stringBuilder.Append(l);
            stringBuilder.Append(ITokenizer.SpecialTokens.EndOfTheSequence);
            
            return stringBuilder.ToString();
        })).ToList(),
        numberOfIterations);

    return model;

}