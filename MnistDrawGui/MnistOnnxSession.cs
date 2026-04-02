using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MnistDrawGui;

/// <summary>
/// Loads an ONNX model exported from the Keras MNIST network (input 28×28×1 float32, NHWC).
/// </summary>
public sealed class MnistOnnxSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly bool _channelsLast;

    public MnistOnnxSession(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);

        var opts = new SessionOptions();
        _session = new InferenceSession(modelPath, opts);

        var meta = _session.InputMetadata.First();
        _inputName = meta.Key;
        var shape = meta.Value.Dimensions.ToArray();

        // Typical Keras CNN: [N, 28, 28, 1] or legacy [N, 1, 28, 28]
        if (shape.Length == 4)
        {
            var last = shape[3] == 1 || shape[3] == -1;
            var c1 = shape[1] == 1 || shape[1] == -1;
            _channelsLast = last || (!c1 && shape[3] > 0);
        }
        else if (shape.Length == 2 && (shape[1] == 784 || shape[1] == -1))
            _channelsLast = false; // flat vector — handled in Run
        else
            _channelsLast = true;
    }

    /// <summary>
    /// Row-major 28×32 values in [0, 1], white digit on black background (MNIST convention).
    /// </summary>
    public float[] Predict(float[,] gray28)
    {
        var meta = _session.InputMetadata[_inputName];
        var dims = meta.Dimensions.ToArray();

        DenseTensor<float> input;
        if (dims.Length == 2)
        {
            var flat = new float[784];
            var i = 0;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                flat[i++] = gray28[y, x];
            input = new DenseTensor<float>(flat, new[] { 1, 784 });
        }
        else if (dims.Length == 4 && _channelsLast)
        {
            var data = new float[28 * 28];
            var i = 0;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                data[i++] = gray28[y, x];
            input = new DenseTensor<float>(data, new[] { 1, 28, 28, 1 });
        }
        else if (dims.Length == 4)
        {
            // NCHW
            var data = new float[28 * 28];
            var i = 0;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                data[i++] = gray28[y, x];
            input = new DenseTensor<float>(data, new[] { 1, 1, 28, 28 });
        }
        else
            throw new InvalidOperationException($"Unsupported ONNX input rank {dims.Length}.");

        using var results = _session.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, input)
        });

        var tensor = results.First(v => v.Name == _session.OutputNames[0]).AsTensor<float>();
        return tensor.ToArray();
    }

    public void Dispose() => _session.Dispose();
}
