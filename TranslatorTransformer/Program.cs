// See https://aka.ms/new-console-template for more information

using TranslatorTransformer.Core.Model.Transformer;
using TranslatorTransformer.Core.Tokenization;
using TranslatorTransformer.Core.Tokenization.BPE;
const int numberOfIterations = 100;

var engText = File.ReadAllText("Translations/EN.txt");
var rusText = File.ReadAllText("Translations/RU.txt");
Console.OutputEncoding = System.Text.Encoding.UTF8;
var tokenizer = new BPETokenizer();
tokenizer.Train(engText + rusText, ITokenizer.VocabSize);
var model = new TransformerInferenceModel(tokenizer);

var engLines = engText.Split('\n');
var rusLines = rusText.Split('\n');

model.Train(
    engLines.Zip(rusLines.Select(l =>
        ITokenizer.SpecialTokens.StartOfTheSequence + l + ITokenizer.SpecialTokens.EndOfTheSequence)).ToList(),
    numberOfIterations);

while (true)
{
    Console.Write("Enter the text: ");
    var sourceText = Console.ReadLine();
    var tokens = model.PerformInference(sourceText, "").ToList();
    
    Console.WriteLine(tokenizer.Decode(tokens));

    Console.WriteLine();
}