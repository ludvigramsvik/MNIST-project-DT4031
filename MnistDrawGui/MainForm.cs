using System.Diagnostics;

namespace MnistDrawGui;

public sealed class MainForm : Form
{
    private readonly DigitCanvas _canvas = new() { Dock = DockStyle.Fill };
    private readonly ProbabilityPanel _probPanel = new() { Dock = DockStyle.Fill };
    private readonly Label _predictionLabel = new();
    private readonly Label _statusLabel = new();
    private readonly System.Windows.Forms.Timer _predictTimer;
    private MnistOnnxSession? _session;
    private volatile bool _predictPending;

    public MainForm()
    {
        Text = "MNIST digit sketch — live prediction";
        StartPosition = FormStartPosition.CenterScreen;
        MinimumSize = new Size(820, 560);
        BackColor = Color.FromArgb(22, 22, 26);
        Font = new Font("Segoe UI", 10f);

        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2,
            BackColor = BackColor,
            Padding = new Padding(12)
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var header = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            BackColor = BackColor,
            AutoSize = true
        };

        var btnClear = new Button
        {
            Text = "Clear",
            AutoSize = true,
            Margin = new Padding(0, 6, 12, 0),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(48, 48, 56),
            ForeColor = Color.White,
            Cursor = Cursors.Hand
        };
        btnClear.FlatAppearance.BorderColor = Color.FromArgb(80, 80, 90);
        btnClear.Click += (_, _) => _canvas.ClearCanvas();

        var btnOpenModel = new Button
        {
            Text = "Load ONNX…",
            AutoSize = true,
            Margin = new Padding(0, 6, 12, 0),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(48, 48, 56),
            ForeColor = Color.White,
            Cursor = Cursors.Hand
        };
        btnOpenModel.FlatAppearance.BorderColor = Color.FromArgb(80, 80, 90);
        btnOpenModel.Click += OpenModel_Click;

        _predictionLabel = new Label
        {
            AutoSize = true,
            Margin = new Padding(12, 10, 0, 0),
            ForeColor = Color.FromArgb(170, 220, 255),
            Font = new Font("Segoe UI Semibold", 16f, FontStyle.Bold),
            Text = "Prediction: —"
        };

        _statusLabel = new Label
        {
            AutoSize = true,
            Margin = new Padding(12, 14, 0, 0),
            ForeColor = Color.Gray,
            Text = ""
        };

        header.Controls.Add(btnClear);
        header.Controls.Add(btnOpenModel);
        header.Controls.Add(_predictionLabel);
        header.Controls.Add(_statusLabel);

        var split = new SplitContainer
        {
            Dock = DockStyle.Fill,
            Orientation = Orientation.Vertical,
            SplitterWidth = 8,
            Panel1MinSize = 320,
            Panel2MinSize = 280,
            SplitterDistance = 480,
            BackColor = Color.FromArgb(35, 35, 42)
        };

        var pad = new Panel { Dock = DockStyle.Fill, Padding = new Padding(0, 0, 8, 0), BackColor = BackColor };
        pad.Controls.Add(_canvas);
        split.Panel1.Controls.Add(pad);

        var right = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            RowCount = 2,
            ColumnCount = 1,
            BackColor = BackColor
        };
        right.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
        right.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        var hint = new Label
        {
            Dock = DockStyle.Fill,
            Text = "Class probabilities (updates while you draw)",
            ForeColor = Color.DarkGray,
            TextAlign = ContentAlignment.BottomLeft,
            Padding = new Padding(0, 0,0, 4)
        };
        right.Controls.Add(hint, 0, 0);
        right.Controls.Add(_probPanel, 0, 1);

        split.Panel2.Controls.Add(right);

        root.Controls.Add(header, 0, 0);
        root.Controls.Add(split, 0, 1);
        Controls.Add(root);

        _predictTimer = new System.Windows.Forms.Timer { Interval = 45 };
        _predictTimer.Tick += PredictTimer_Tick;

        Load += MainForm_Load;

        _canvas.StrokesChanged += (_, _) =>
        {
            if (_session != null)
                _predictPending = true;
        };
    }

    private void MainForm_Load(object? sender, EventArgs e)
    {
        TryLoadDefaultModel();
        _predictTimer.Start();
    }

    private void TryLoadDefaultModel()
    {
        var dir = AppContext.BaseDirectory;
        var path = Path.Combine(dir, "mnist_model.onnx");
        if (!File.Exists(path))
        {
            var root = Path.GetFullPath(Path.Combine(dir, "..", "..", "..", ".."));
            var fromRepo = Path.Combine(root, "MnistDrawGui", "mnist_model.onnx");
            if (File.Exists(fromRepo))
                path = fromRepo;
        }

        if (File.Exists(path))
            LoadModel(path);
        else
            SetStatus("Place mnist_model.onnx next to the app or use Export (see NN/export_mnist_to_onnx.py).");
    }

    private void LoadModel(string path)
    {
        try
        {
            _session?.Dispose();
            _session = new MnistOnnxSession(path);
            SetStatus(Path.GetFileName(path));
            _predictPending = true;
        }
        catch (Exception ex)
        {
            _session = null;
            SetStatus("Could not load ONNX: " + ex.Message);
        }
    }

    private void SetStatus(string msg) => _statusLabel.Text = msg;

    private void OpenModel_Click(object? sender, EventArgs e)
    {
        using var dlg = new OpenFileDialog
        {
            Filter = "ONNX model|*.onnx|All files|*.*",
            Title = "Select mnist_model.onnx"
        };
        if (dlg.ShowDialog(this) == DialogResult.OK)
            LoadModel(dlg.FileName);
    }

    private void PredictTimer_Tick(object? sender, EventArgs e)
    {
        if (_session == null || !_predictPending)
            return;
        _predictPending = false;

        try
        {
            var grid = _canvas.ToMnistGrid28();
            var maxLum = 0f;
            for (var y = 0; y < 28; y++)
            for (var x = 0; x < 28; x++)
                if (grid[y, x] > maxLum)
                    maxLum = grid[y, x];

            if (maxLum < 0.11f)
            {
                _predictionLabel.Text = "Prediction: —";
                _probPanel.SetProbabilities(new float[10]);
                return;
            }

            var raw = _session.Predict(grid);
            if (raw.Length != 10)
            {
                SetStatus($"Model output length {raw.Length}; expected 10.");
                return;
            }

            _probPanel.SetProbabilities(raw);

            var sum = raw.Sum();
            var likeProbs = sum is >= 0.92f and <= 1.08f;
            float[] pr;
            if (likeProbs)
            {
                pr = new float[10];
                for (var i = 0; i < 10; i++)
                    pr[i] = Math.Clamp(raw[i], 0f, 1f);
                var s = pr.Sum();
                if (s > 1e-6f)
                    for (var i = 0; i < 10; i++)
                        pr[i] /= s;
            }
            else
            {
                var sumExp = 0f;
                pr = new float[10];
                for (var i = 0; i < 10; i++)
                {
                    pr[i] = MathF.Exp(Math.Clamp(raw[i], -50f, 50f));
                    sumExp += pr[i];
                }
                if (sumExp > 1e-6f)
                    for (var i = 0; i < 10; i++)
                        pr[i] /= sumExp;
            }

            var best = 0;
            for (var i = 1; i < 10; i++)
                if (pr[i] > pr[best])
                    best = i;

            _predictionLabel.Text = $"Prediction:  {best}   ({pr[best] * 100:F1}%)";
        }
        catch (Exception ex)
        {
            Debug.WriteLine(ex);
            SetStatus("Inference error: " + ex.Message);
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _predictTimer.Dispose();
            _session?.Dispose();
        }
        base.Dispose(disposing);
    }
}
