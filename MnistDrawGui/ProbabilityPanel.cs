namespace MnistDrawGui;

/// <summary>
/// Renders class probabilities as horizontal bars (0–9).
/// </summary>
public sealed class ProbabilityPanel : Control
{
    private float[] _probs = new float[10];

    public ProbabilityPanel()
    {
        DoubleBuffered = true;
        SetStyle(ControlStyles.ResizeRedraw, true);
        BackColor = Color.FromArgb(28, 28, 34);
        ForeColor = Color.Gainsboro;
    }

    public void SetProbabilities(float[] logitsOrProbs)
    {
        if (logitsOrProbs.Length != 10)
            return;

        var sum = 0f;
        var min = float.MaxValue;
        var max = float.MinValue;
        for (var i = 0; i < 10; i++)
        {
            var v = logitsOrProbs[i];
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        var likeProbs = sum is >= 0.92f and <= 1.08f && min >= -0.05f && max <= 1.05f;
        if (likeProbs)
        {
            for (var i = 0; i < 10; i++)
                _probs[i] = Math.Clamp(logitsOrProbs[i], 0f, 1f);
            var s = _probs.Sum();
            if (s > 1e-6f)
                for (var i = 0; i < 10; i++)
                    _probs[i] /= s;
        }
        else
        {
            var sumExp = 0f;
            for (var i = 0; i < 10; i++)
            {
                _probs[i] = MathF.Exp(Math.Clamp(logitsOrProbs[i], -50f, 50f));
                sumExp += _probs[i];
            }
            if (sumExp > 1e-6f)
                for (var i = 0; i < 10; i++)
                    _probs[i] /= sumExp;
        }

        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var g = e.Graphics;
        g.Clear(BackColor);

        var pad = 8;
        var rowH = Math.Max(14, (ClientSize.Height - pad * 2) / 10 - 4);
        var labelW = 22;
        var barX = pad + labelW;
        var barW = ClientSize.Width - barX - pad;
        var y0 = pad;

        using var font = new Font(Font.FontFamily, Math.Clamp(rowH * 0.45f, 8f, 14f), FontStyle.Regular);
        using var brushText = new SolidBrush(ForeColor);
        using var brushBarBg = new SolidBrush(Color.FromArgb(50, 55, 65));
        var accent = Color.FromArgb(120, 200, 255);

        for (var d = 0; d < 10; d++)
        {
            var y = y0 + d * (rowH + 4);
            g.DrawString(d.ToString(), font, brushText, pad, y + rowH * 0.15f);

            g.FillRectangle(brushBarBg, barX, y, barW, rowH);

            var w = barW * _probs[d];
            using var brushBar = new SolidBrush(Color.FromArgb(
                (int)(accent.R * 0.4 + 255 * 0.6 * _probs[d]),
                (int)(accent.G * 0.4 + 255 * 0.6 * _probs[d]),
                (int)(accent.B * 0.4 + 255 * 0.6 * _probs[d])));
            g.FillRectangle(brushBar, barX, y, Math.Max(1, w), rowH);

            var pct = $"{_probs[d] * 100:F1}%";
            g.DrawString(pct, font, brushText, barX + barW - 48, y + rowH * 0.1f);
        }
    }
}
