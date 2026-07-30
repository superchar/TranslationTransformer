using FluentAssertions;
using TranslatorTransformer.Core.Model.Transformer;
using TranslatorTransformer.Core.Tokenization;
using TranslatorTransformer.Core.Tokenization.BPE;

namespace TranslatorTransformer.Tests.Model.Transformer;

public class TransformerInferenceModelTests
{
    private const string StartSequenceToken = ITokenizer.SpecialTokens.StartOfTheSequence;
    private const string EndSequenceToken = ITokenizer.SpecialTokens.EndOfTheSequence;
    private  const int NumberOfIterations = 10;
    private const int VocabSize = 300;
    
    private readonly BPETokenizer _tokenizer;
    private readonly TransformerInferenceModel _model;

    public TransformerInferenceModelTests()
    {
        _tokenizer = new BPETokenizer();
        _model = new TransformerInferenceModel(_tokenizer);
    }

    [Theory]
    [MemberData(nameof(GetTrainingDocuments))]
    public void Train_NeverOutputsSpecialTokens(string sourceText, string targetText)
    {
        _tokenizer.Train(sourceText + targetText, VocabSize);

        _model.Train([(sourceText, targetText)], NumberOfIterations);

        var modelOutputTokens = _model.PerformInference(sourceText, string.Empty).ToList();
        var resultText = _tokenizer.Decode(modelOutputTokens);

        ITokenizer.SpecialTokens.All.All(token => !resultText.Contains(token)).Should().BeTrue();
    }
    
    [Fact]
    public void Train_HandlesBatchesCorrectly()
    {
        const int numberOfIterations = 50;
        var batch = GetDocumentsBatch().ToList();
        var aggregatedContent = batch.Aggregate(string.Empty, (acc, item) => acc + item.SourceText + item.TargetText);
        _tokenizer.Train(aggregatedContent, VocabSize);

        _model.Train(batch, numberOfIterations);

        foreach (var (sourceText, targetText) in batch)
        {
            var modelOutputTokens = _model.PerformInference(sourceText, string.Empty).ToList();
            var resultText = _tokenizer.Decode(modelOutputTokens);
            targetText.Should().Contain(resultText);
        }
        
    }
    
    [Theory] 
    [MemberData(nameof(GetTrainingDocuments))]
    public void Train_OvertrainingOnOneBatch_PredictsTranslationCorrectly(string sourceText, string targetText)
    {
        _tokenizer.Train(sourceText + targetText, VocabSize);

        _model.Train([(sourceText, targetText)], NumberOfIterations);

        var modelOutputTokens = _model.PerformInference(sourceText, string.Empty).ToList();
        var resultText = _tokenizer.Decode(modelOutputTokens);
        
        targetText.Should().Contain(resultText);
    }

    public static IEnumerable<object[]> GetTrainingDocuments()
    {
        foreach (var (sourceText, targetText) in GetDocumentsBatch())
        {
            yield return [sourceText, targetText];
        }
    }

    private static IEnumerable<(string SourceText, string TargetText)> GetDocumentsBatch()
    {
        yield return
        (
            $"{StartSequenceToken}Hello, my name is Vlad{EndSequenceToken}",
            $"{StartSequenceToken}Привет, меня зовут Влад{EndSequenceToken}"
        );
        yield return
        (
            $"{StartSequenceToken}I love walking.!!!{EndSequenceToken}",
            $"{StartSequenceToken}Я люблю гулять.!!!{EndSequenceToken}"
        );
        yield return
        (
            $"{StartSequenceToken}The weather is lovely today, isn't it?{EndSequenceToken}",
            $"{StartSequenceToken}Сегодня отличная погода, не правда ли?{EndSequenceToken}"
        );
    }
}