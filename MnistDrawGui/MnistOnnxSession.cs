using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MnistDrawGui;

public sealed class FullPrediction
{
    public required float[] ClassOutput { get; init; }
    public required IReadOnlyList<ActivationColumn> LayerColumns { get; init; }
}

/// <summary>
/// Loads ONNX from Keras export; supports multi-output graphs for layer visualisation.
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

        if (shape.Length == 4)
        {
            var last = shape[3] == 1 || shape[3] == -1;
            var c1 = shape[1] == 1 || shape[1] == -1;
            _channelsLast = last || (!c1 && shape[3] > 0);
        }
        else if (shape.Length == 2 && (shape[1] == 784 || shape[1] == -1))
            _channelsLast = false;
        else
            _channelsLast = true;
    }

    public float[] Predict(float[,] gray28) => PredictFull(gray28).ClassOutput;

    public FullPrediction PredictFull(float[,] gray28)
    {
        var input = BuildTensor(gray28);
        using var results = _session.Run(new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, input)
        });

        var outputs = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (var r in results)
            outputs[r.Name] = r.AsTensor<float>().ToArray();

        var (classOut, classKey) = ResolveClassOutput(outputs);
        var columns = BuildLayerColumns(outputs, classOut, classKey);

        return new FullPrediction
        {
            ClassOutput = classOut,
            LayerColumns = columns
        };
    }

    private static (float[] Values, string Key) ResolveClassOutput(Dictionary<string, float[]> outputs)
    {
        foreach (var kv in outputs)
        {
            if (kv.Value.Length != 10)
                continue;
            if (string.Equals(kv.Key, "output", StringComparison.OrdinalIgnoreCase)
                || kv.Key.EndsWith("_output", StringComparison.OrdinalIgnoreCase))
                return (kv.Value, kv.Key);
        }

        foreach (var kv in outputs)
            if (kv.Value.Length == 10)
                return (kv.Value, kv.Key);

        throw new InvalidOperationException("No 10-class output found in ONNX model.");
    }

    private static List<ActivationColumn> BuildLayerColumns(
        Dictionary<string, float[]> outputs,
        float[] classOut,
        string classKey)
    {
        var list = new List<ActivationColumn>();
        foreach (var kv in outputs.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            if (kv.Key == classKey)
                continue;
            if (kv.Value.Length == 0)
                continue;
            list.Add(new ActivationColumn(PrettyName(kv.Key), kv.Value));
        }

        list.Add(new ActivationColumn("Output", (float[])classOut.Clone()));
        return list;
    }

    private static string PrettyName(string onnxName)
    {
        var s = onnxName;
        if (s.Length > 3 && char.IsDigit(s[0]) && char.IsDigit(s[1]) && s[2] == '_')
            s = s[3..];
        if (s.EndsWith("_output", StringComparison.OrdinalIgnoreCase))
            s = s[..^7];
        return s.Replace('_', ' ').Trim();
    }

    private DenseTensor<float> BuildTensor(float[,] gray28)
    {
        var meta = _session.InputMetadata[_inputName];
        var dims = meta.Dimensions.ToArray();

        if (dims.Length == 2)
        {
            var flat = new float[784];
            var i = 0;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                flat[i++] = gray28[y, x];
            return new DenseTensor<float>(flat, new[] { 1, 784 });
        }

        if (dims.Length == 4 && _channelsLast)
        {
            var data = new float[28 * 28];
            var i = 0;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                data[i++] = gray28[y, x];
            return new DenseTensor<float>(data, new[] { 1, 28, 28, 1 });
        }

        if (dims.Length == 4)
        {
            var data = new float[28 * 28];
            var i = 0;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                data[i++] = gray28[y, x];
            return new DenseTensor<float>(data, new[] { 1, 1, 28, 28 });
        }

        throw new InvalidOperationException($"Unsupported ONNX input rank {dims.Length}.");
    }

    public void Dispose() => _session.Dispose();
}
