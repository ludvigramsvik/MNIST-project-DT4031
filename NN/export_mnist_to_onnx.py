r"""
Export the trained Keras MNIST model to ONNX for the WinForms app (MnistDrawGui).

Prerequisites:
  pip install tensorflow tf2onnx onnx

Usage:
  python export_mnist_to_onnx.py
  python export_mnist_to_onnx.py --input path\to\model.keras

The default file name is looked up in this order:
  1) current working directory
  2) this script's folder (NN/, same as Model_Implementation.ipynb)
  3) parent folder (repo root)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile


def _resolve_keras_path(user_path: str) -> str:
    """Find .keras file whether you run from repo root or NN/."""
    if os.path.isabs(user_path) and os.path.isfile(user_path):
        return os.path.abspath(user_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tries = [
        os.path.abspath(user_path),
        os.path.normpath(os.path.join(script_dir, os.path.basename(user_path))),
        os.path.normpath(os.path.join(script_dir, user_path)),
        os.path.normpath(os.path.join(script_dir, "..", os.path.basename(user_path))),
    ]
    seen: set[str] = set()
    for t in tries:
        if t in seen:
            continue
        seen.add(t)
        if os.path.isfile(t):
            return t

    msg = (
        "Model not found.\n\n"
        "Searched:\n  "
        + "\n  ".join(seen)
        + "\n\n"
        "The trained file is usually next to your notebooks:\n"
        "  NN\\mnist_model_modified2_data.keras\n\n"
        "If you only have it elsewhere, run:\n"
        "  python export_mnist_to_onnx.py --input \"C:\\path\\to\\mnist_model_modified2_data.keras\""
    )
    raise SystemExit(msg)


def _keras_to_onnx(model: object, spec: tuple, opset: int):
    """Try export paths that work with Keras 3 / TF 2.16+ (avoids KeyError: keras_tensor_*)."""
    import tensorflow as tf
    import tf2onnx

    def wrap(m) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(28, 28, 1), name="input", dtype=tf.float32)
        out = m(inp, training=False)
        return tf.keras.Model(inputs=inp, outputs=out, name="mnist_onnx_export")

    wrapped = wrap(model)

    try:
        return tf2onnx.convert.from_keras(wrapped, input_signature=spec, opset=opset)
    except KeyError:
        pass

    tmp = tempfile.mkdtemp(suffix="_sm")
    try:
        tf.saved_model.save(wrapped, tmp)
        return tf2onnx.convert.from_saved_model(
            tmp,
            opset=opset,
            signature_def="serving_default",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Keras MNIST model to ONNX (float NHWC).")
    parser.add_argument(
        "--input",
        default="mnist_model_modified2_data.keras",
        help="Path or filename of the saved Keras model (.keras).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "MnistDrawGui", "mnist_model.onnx"),
        help="Output .onnx path.",
    )
    args = parser.parse_args()

    in_path = _resolve_keras_path(args.input)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    import tensorflow as tf

    print(f"Loading Keras model: {in_path}")
    model = tf.keras.models.load_model(in_path)
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

    model_proto, _ = _keras_to_onnx(model, spec, opset=13)
    with open(out_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
