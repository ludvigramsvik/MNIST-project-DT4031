using System.Drawing.Drawing2D;

namespace MnistDrawGui;

/// <summary>
/// High-DPI-friendly drawing surface: black background, soft white strokes (MNIST-like).
/// </summary>
public sealed class DigitCanvas : Control
{
    private Bitmap? _buffer;
    private Point? _lastPoint;
    private bool _drawing;
    private readonly Color _paper = Color.FromArgb(18, 18, 22);
    private readonly Color _ink = Color.FromArgb(245, 245, 250);

    public event EventHandler? StrokesChanged;

    public DigitCanvas()
    {
        DoubleBuffered = true;
        SetStyle(ControlStyles.ResizeRedraw, true);
        TabStop = false;
        BackColor = _paper;
        MinimumSize = new Size(120, 120);
    }

    private float StrokeWidth => Math.Clamp(Math.Min(ClientSize.Width, ClientSize.Height) / 14f, 10f, 36f);

    protected override void OnSizeChanged(EventArgs e)
    {
        base.OnSizeChanged(e);
        EnsureBuffer();
    }

    private void EnsureBuffer()
    {
        if (ClientSize.Width <= 0 || ClientSize.Height <= 0)
            return;

        if (_buffer is { Width: var w, Height: var h } && w == ClientSize.Width && h == ClientSize.Height)
            return;

        _buffer?.Dispose();
        _buffer = new Bitmap(ClientSize.Width, ClientSize.Height, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
        using var g = Graphics.FromImage(_buffer);
        g.Clear(_paper);
    }

    public void ClearCanvas()
    {
        EnsureBuffer();
        if (_buffer == null) return;
        using var g = Graphics.FromImage(_buffer);
        g.Clear(_paper);
        Invalidate();
        StrokesChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    /// Downsamples the canvas to 28×28, luminance in [0,1] (white stroke on dark bg).
    /// </summary>
    public float[,] ToMnistGrid28()
    {
        EnsureBuffer();
        if (_buffer == null)
            return new float[28, 28];

        using var small = new Bitmap(28, 28, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
        using (var g = Graphics.FromImage(small))
        {
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.CompositingQuality = CompositingQuality.HighQuality;
            g.SmoothingMode = SmoothingMode.None;
            g.DrawImage(_buffer, new Rectangle(0, 0, 28, 28));
        }

        var grid = new float[28, 28];
        for (var y = 0; y < 28; y++)
        for (var x = 0; x < 28; x++)
        {
            var c = small.GetPixel(x, y);
            // perceived luminance
            var lum = (0.299f * c.R + 0.587f * c.G + 0.114f * c.B) / 255f;
            grid[y, x] = lum;
        }

        return grid;
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        EnsureBuffer();
        if (_buffer != null)
            e.Graphics.DrawImage(_buffer, Point.Empty);
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button != MouseButtons.Left) return;
        Focus();
        EnsureBuffer();
        _drawing = true;
        _lastPoint = e.Location;
        DrawDot(e.Location);
        Invalidate();
        StrokesChanged?.Invoke(this, EventArgs.Empty);
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (!_drawing || !_lastPoint.HasValue || _buffer == null) return;
        if (e.Button != MouseButtons.Left)
        {
            _drawing = false;
            _lastPoint = null;
            return;
        }

        using var g = Graphics.FromImage(_buffer);
        g.SmoothingMode = SmoothingMode.AntiAlias;
        g.CompositingQuality = CompositingQuality.HighQuality;
        using var pen = new Pen(_ink, StrokeWidth)
        {
            StartCap = LineCap.Round,
            EndCap = LineCap.Round,
            LineJoin = LineJoin.Round
        };
        g.DrawLine(pen, _lastPoint.Value, e.Location);
        _lastPoint = e.Location;
        Invalidate();
        StrokesChanged?.Invoke(this, EventArgs.Empty);
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        _drawing = false;
        _lastPoint = null;
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);
        _drawing = false;
        _lastPoint = null;
    }

    private void DrawDot(Point p)
    {
        if (_buffer == null) return;
        using var g = Graphics.FromImage(_buffer);
        g.SmoothingMode = SmoothingMode.AntiAlias;
        var r = StrokeWidth / 2f;
        using var brush = new SolidBrush(_ink);
        g.FillEllipse(brush, p.X - r, p.Y - r, StrokeWidth, StrokeWidth);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _buffer?.Dispose();
        base.Dispose(disposing);
    }
}
