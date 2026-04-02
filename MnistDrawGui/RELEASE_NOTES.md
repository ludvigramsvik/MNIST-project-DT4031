## MNIST Digit Sketch (WinForms)

**Draw digits with the mouse and get live ONNX predictions** — no button press, with a probability chart for all ten classes.

### Included in the zip

- `MnistDrawGui.exe` and dependencies (from `dotnet publish`)
- Add **`mnist_model.onnx`** next to the exe if it is not already in the archive (or export it with `NN/export_mnist_to_onnx.py`)

### Before you run

- **Windows 64-bit**
- **[.NET 8 Desktop Runtime (x64)](https://dotnet.microsoft.com/download/dotnet/8.0)** — *skip this if the download is marked **self-contained***

### Usage

Unzip → ensure `mnist_model.onnx` is beside the `.exe` → run `MnistDrawGui.exe` → draw → **Clear** to reset.

Full details: [`MnistDrawGui/README.md`](./README.md).
