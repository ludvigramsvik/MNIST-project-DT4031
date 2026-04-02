# MNIST Digit Sketch — WinForms

Draw digits with your mouse and see **live** neural-network predictions. This desktop app loads your trained MNIST model as **ONNX** and runs it with [ONNX Runtime](https://onnxruntime.ai/) — no Python required to use the published build.

---

## Highlights

- **Smooth drawing** — Anti-aliased strokes on a **black** canvas so preprocessed input matches MNIST-style images (dark background, bright digit).
- **Real-time inference** — Probabilities and the top digit update continuously while you draw; no “Predict” button.
- **Clear layout** — Large sketch area plus a bar chart of class scores (0–9).
- **Flexible model loading** — Ships with `mnist_model.onnx` next to the executable when present, or use **Load ONNX…** to pick any compatible file.

---

## Requirements (prebuilt `publish` folder)

| Requirement | Notes |
|-------------|--------|
| **Windows 10/11 (64-bit)** | Build target: `win-x64`. |
| **[.NET 8 Desktop Runtime (x64)](https://dotnet.microsoft.com/download/dotnet/8.0)** | Only for *framework-dependent* publishes (`--self-contained false`). |
| **`mnist_model.onnx`** | Same folder as `MnistDrawGui.exe`, unless you load a model via the UI. |

**Self-contained publish:** If you distribute a build created with `--self-contained true`, recipients **do not** need to install the .NET runtime (the folder is larger).

On very clean machines, if native libraries fail to load, install the **[Visual C++ Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)**.

---

## Quick start

1. Unzip the release archive (or use your `publish` output folder).
2. Ensure **`mnist_model.onnx`** sits beside **`MnistDrawGui.exe`** (included in releases when we attach it, or export it yourself — see below).
3. Double-click **`MnistDrawGui.exe`**.
4. Draw a digit; watch the prediction and probability bars update as you draw. **Clear** resets the canvas.

---

## Building from source

```powershell
cd MnistDrawGui
dotnet publish -c Release -r win-x64 --self-contained false -o .\publish
```

Output: **`MnistDrawGui\publish\`**

Copy **`mnist_model.onnx`** into `publish` if it was not copied automatically (the project copies it when the file exists next to the `.csproj`).

---

## Exporting `mnist_model.onnx` from Keras

The WinForms app does not read `.keras` files directly. One-time export from the repo:

```powershell
cd NN
pip install tensorflow tf2onnx onnx
python export_mnist_to_onnx.py
```

By default this writes **`../MnistDrawGui/mnist_model.onnx`**. Use `--input` / `--output` if your paths differ.

---

## Troubleshooting

| Issue | What to try |
|--------|-------------|
| App won’t start | Install **.NET 8 Desktop Runtime (x64)** for framework-dependent builds. |
| “Could not load ONNX” / no predictions | Place **`mnist_model.onnx`** next to the `.exe` or use **Load ONNX…**. |
| Poor accuracy on drawn digits | Normal *domain gap* vs scanned MNIST; draw larger, centered strokes. Further improvements can use bbox centering in preprocessing (see project issues / PRs). |

---

## Repository layout (related files)

| Path | Role |
|------|------|
| `MnistDrawGui/` | C# WinForms source and project file |
| `NN/export_mnist_to_onnx.py` | Keras → ONNX export |
| `NN/Model_Implementation.ipynb` | Model training / evaluation |

---

## License

Use the same license as the parent **MNIST-project-DT4031** repository (add a `LICENSE` file at the repo root if none exists yet).

---

*Part of the MNIST course / project — DT4031.*
