// See https://aka.ms/new-console-template for more information

using TranslatorTransformer.Core.Model.Transformer;
using TranslatorTransformer.Core.Tokenization;
using TranslatorTransformer.Core.Tokenization.BPE;

var engText = File.ReadAllText("Translations/EN.txt");
var rusText = File.ReadAllText("Translations/RU.txt");

var tokenizer = new BPETokenizer();
tokenizer.Train(engText + rusText, ITokenizer.VocabSize);
var model = new TransformerInferenceModel(tokenizer);

var engLines = engText.Split('\n');
var rusLines = rusText.Split('\n');

model.Train(engLines.Zip(rusLines));

var result = model.PerformInference("The cosmos began not with a bang", "");

foreach (var item in result)
{
    Console.Write(item);
}