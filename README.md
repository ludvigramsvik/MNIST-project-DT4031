# MNIST-project-DT4031

DT4031 coursework: MNIST with neural networks, SVM, visualizations, and a small **Windows** sketch app for live digit classification.

## Releases

Prebuilt **MnistDrawGui** builds (zip of your `publish` folder) live on GitHub:

- **[All releases](https://github.com/ludvigramsvik/MNIST-project-DT4031/releases)**  
- **[Latest release](https://github.com/ludvigramsvik/MNIST-project-DT4031/releases/latest)** — download the attached `.zip`, unzip, run `MnistDrawGui.exe` (see requirements below).

*Creating a release:* on GitHub go to **Releases → Draft a new release**, choose a tag (e.g. `v1.0.0`), paste notes from [`MnistDrawGui/RELEASE_NOTES.md`](MnistDrawGui/RELEASE_NOTES.md), and upload your zip as a release asset.

More detail: [`MnistDrawGui/README.md`](MnistDrawGui/README.md).

---

## MNIST Digit Sketch (WinForms)

Draw digits with your mouse and see **live** neural-network predictions. This desktop app loads your trained MNIST model as **ONNX** and runs it with [ONNX Runtime](https://onnxruntime.ai/) — no Python required to use the published build.

### Highlights

- **Smooth drawing** — Anti-aliased strokes on a **black** canvas so preprocessed input matches MNIST-style images (dark background, bright digit).
- **Real-time inference** — Probabilities and the top digit update continuously while you draw; no “Predict” button.
- **Clear layout** — Large sketch area plus a bar chart of class scores (0–9).
- **Flexible model loading** — Ships with `mnist_model.onnx` next to the executable when present, or use **Load ONNX…** to pick any compatible file.

---

### Requirements (prebuilt `publish` folder)

| Requirement | Notes |
|-------------|--------|
| **Windows 10/11 (64-bit)** | Build target: `win-x64`. |
| **[.NET 8 Desktop Runtime (x64)](https://dotnet.microsoft.com/download/dotnet/8.0)** | Only for *framework-dependent* publishes (`--self-contained false`). |
| **`mnist_model.onnx`** | Same folder as `MnistDrawGui.exe`, unless you load a model via the UI. |

**Self-contained publish:** If you distribute a build created with `--self-contained true`, recipients **do not** need to install the .NET runtime (the folder is larger).

On very clean machines, if native libraries fail to load, install the **[Visual C++ Redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)**.

---
