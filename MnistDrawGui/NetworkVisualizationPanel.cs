namespace MnistDrawGui;

/// <summary>
/// Draws columns of "neurons" (circles) whose brightness follows activation strength.
/// </summary>
public sealed class NetworkVisualizationPanel : Control
{
    private float[] _inputPreview = Array.Empty<float>();
    private List<ActivationColumn> _columns = new();
    private const int MaxNodesPerColumn = 56;
    private readonly StringFormat _sfCenter = new() { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center };

    public NetworkVisualizationPanel()
    {
        DoubleBuffered = true;
        SetStyle(ControlStyles.ResizeRedraw, true);
        BackColor = Color.FromArgb(24, 24, 30);
        ForeColor = Color.Gainsboro;
        MinimumSize = new Size(180, 120);
    }

    public void Clear()
    {
        _inputPreview = Array.Empty<float>();
        _columns.Clear();
        Invalidate();
    }

    /// <summary>
    /// Left-to-right: optional input column (downsampled pixels), then hidden layers, then output.
    /// </summary>
    public void SetActivations(float[] inputDownsampled, IReadOnlyList<ActivationColumn> columns)
    {
        _inputPreview = (float[])inputDownsampled.Clone();
        _columns = columns.Select(c => new ActivationColumn(c.Title, (float[])c.Values.Clone())).ToList();
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var g = e.Graphics;
        g.Clear(BackColor);
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

        if (_columns.Count == 0 && _inputPreview.Length == 0)
        {
            using var hintFont = new Font(Font.FontFamily, 8.5f);
            using var br = new SolidBrush(Color.FromArgb(100, 100, 110));
            g.DrawString(
                "Re-export with NN/export_mnist_to_onnx.py for hidden-layer activations.\nOutput neurons still light up from predictions.",
                hintFont,
                br,
                ClientRectangle,
                new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center });
            return;
        }

        var pad = 8;
        var colDefs = new List<(string Title, float[] Vals, bool IsInput)>();
        if (_inputPreview.Length > 0)
            colDefs.Add(("Input", _inputPreview, true));
        foreach (var c in _columns)
            colDefs.Add((c.Title, c.Values, false));

        if (colDefs.Count == 0)
            return;

        var colW = (ClientSize.Width - pad * 2) / (float)colDefs.Count;
        using var titleFont = new Font(Font.FontFamily, 7.5f, FontStyle.Bold);
        using var titleBrush = new SolidBrush(Color.FromArgb(160, 170, 190));

        for (var ci = 0; ci < colDefs.Count; ci++)
        {
            var (title, vals, isInput) = colDefs[ci];
            var x0 = pad + ci * colW;
            var colRect = new RectangleF(x0, pad, colW - 4, ClientSize.Height - pad * 2);
            g.DrawString(title, titleFont, titleBrush, new RectangleF(colRect.X, colRect.Y, colRect.Width, 14), _sfCenter);

            var bodyTop = colRect.Y + 18;
            var bodyH = colRect.Height - 20;
            var body = new RectangleF(colRect.X, bodyTop, colRect.Width, bodyH);

            var n = vals.Length;
            var show = Math.Min(MaxNodesPerColumn, n);
            var idx = SubsampleIndices(n, show);
            var norm = Normalize(vals, idx);

            var cols = Math.Max(1, (int)Math.Sqrt(show * (isInput ? 1.2 : 1)));
            var rows = (int)Math.Ceiling(show / (float)cols);
            var cellW = body.Width / cols;
            var cellH = body.Height / Math.Max(rows, 1);
            var rMax = Math.Min(cellW, cellH) * 0.38f;

            for (var i = 0; i < show; i++)
            {
                var cx = body.X + (i % cols + 0.5f) * cellW;
                var cy = body.Y + (i / cols + 0.5f) * cellH;
                var t = norm[i];
                var dim = Color.FromArgb(35, 38, 48);
                var lit = Color.FromArgb(90, 200, 255);
                var c = Color.FromArgb(
                    (int)(dim.R + (lit.R - dim.R) * t),
                    (int)(dim.G + (lit.G - dim.G) * t),
                    (int)(dim.B + (lit.B - dim.B) * t));
                using var brush = new SolidBrush(c);
                using var pen = new Pen(Color.FromArgb(60, 70, 85), 1f);
                g.FillEllipse(brush, cx - rMax, cy - rMax, rMax * 2, rMax * 2);
                g.DrawEllipse(pen, cx - rMax, cy - rMax, rMax * 2, rMax * 2);
            }
        }
    }

    private static int[] SubsampleIndices(int n, int show)
    {
        if (n <= 0 || show <= 0)
            return Array.Empty<int>();
        if (n <= show)
        {
            var a = new int[n];
            for (var i = 0; i < n; i++) a[i] = i;
            return a;
        }
        var r = new int[show];
        for (var i = 0; i < show; i++)
            r[i] = (int)(i * (n - 1f) / Math.Max(show - 1, 1));
        return r;
    }

    private static float[] Normalize(float[] vals, int[] idx)
    {
        var m = 1e-6f;
        for (var i = 0; i < idx.Length; i++)
        {
            var v = Math.Abs(vals[idx[i]]);
            if (v > m) m = v;
        }
        var o = new float[idx.Length];
        for (var i = 0; i < idx.Length; i++)
            o[i] = Math.Clamp(Math.Abs(vals[idx[i]]) / m, 0f, 1f);
        return o;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _sfCenter.Dispose();
        base.Dispose(disposing);
    }
}

public sealed record ActivationColumn(string Title, float[] Values);
