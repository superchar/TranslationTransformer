using TorchSharp.Modules;
using TranslatorTransformer.Core.Tokenization;
using static TorchSharp.torch;
using static TranslatorTransformer.Core.Model.Transformer.Configuration.ModelConfiguration;

namespace TranslatorTransformer.Core.Model.Transformer;


internal class TransformerEncoderDecoder : nn.Module<Tensor, Tensor, long, Tensor>
{
    private readonly Linear _outputLinear = nn.Linear(HiddenSize, ITokenizer.VocabSize);
    
    private readonly TransformerEncoder _encoder = new();
    private readonly Embedding _encoderPositionalEmbedding = nn.Embedding(MaxContextSize, HiddenSize);
    private readonly Embedding _encoderTokenEmbedding = nn.Embedding(ITokenizer.VocabSize, HiddenSize);
    private readonly LayerNorm _encoderLayerNorm = nn.LayerNorm(HiddenSize);

    private readonly TransformerDecoder _decoder = new();
    private readonly Embedding _decoderPositionalEmbedding = nn.Embedding(MaxContextSize, HiddenSize);
    private readonly Embedding _decoderTokenEmbedding = nn.Embedding(ITokenizer.VocabSize, HiddenSize);
    private readonly LayerNorm _decoderLayerNorm = nn.LayerNorm(HiddenSize);

    public TransformerEncoderDecoder() : base("Encoder decoder transformer") => RegisterComponents();
    public override Tensor forward(Tensor encoderInput, Tensor decoderInput, long paddingTokenId)
    {
        var sourcePaddingMask = encoderInput.eq(paddingTokenId).unsqueeze(1).unsqueeze(2);
        var targetPaddingMask = decoderInput.eq(paddingTokenId).unsqueeze(1).unsqueeze(2);
        
        var encoderSequenceLength = encoderInput.shape[1];
        var encoderPositions =
            arange(encoderSequenceLength, device: encoderInput.device, dtype: ScalarType.Int64)
                .unsqueeze(0);
        var encoderEmbedding = _encoderPositionalEmbedding.forward(encoderPositions) +
                               _encoderTokenEmbedding.forward(encoderInput);
        var encoderOutput = _encoder.forward(encoderEmbedding, sourcePaddingMask);
        encoderOutput = _encoderLayerNorm.forward(encoderOutput);

        var decodeSequenceLength = decoderInput.shape[1];
        var decoderPositions =
            arange(decodeSequenceLength, device: decoderInput.device, dtype: ScalarType.Int64)
                .unsqueeze(0);
        var decoderEmbedding = _decoderPositionalEmbedding.forward(decoderPositions) +
                               _decoderTokenEmbedding.forward(decoderInput);
        var decoderOutput = _decoder.forward(decoderEmbedding, encoderOutput, targetPaddingMask, sourcePaddingMask);
        decoderOutput = _decoderLayerNorm.forward(decoderOutput);
        
        return _outputLinear.forward(decoderOutput);
    }
}