// See https://aka.ms/new-console-template for more information

using TranslatorTransformer.Core.Model.Transformer;
using TranslatorTransformer.Core.Tokenization;
using TranslatorTransformer.Core.Tokenization.BPE;

const string SourceText =
    ITokenizer.SpecialTokens.StartOfTheSequence + "The cosmos began not with a bang, but with a silent line of code. 2. The initial parameters were set to absolute zero. 3. A single variable floated in the vast emptiness of the digital void. ";
const string TargetText =
    ITokenizer.SpecialTokens.StartOfTheSequence + "Космос начался не с большого взрыва, а с безмолвной строки кода. 2. Начальные параметры были установлены на абсолютный ноль. 3. Единственная переменная парила в огромной пустоте цифрового вакуума.";
var tokenizer = new BPETokenizer();
tokenizer.Train([SourceText + TargetText], ITokenizer.VocabSize);
var model = new TransformerInferenceModel(tokenizer);

model.Train([(SourceText, TargetText)]);

var result = model.PerformInference("The cosmos began not with a bang", "");

foreach (var item in result)
{
    Console.Write(item);
}