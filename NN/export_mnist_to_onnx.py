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

    try:
        return tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
    except KeyError:
        pass

    tmp = tempfile.mkdtemp(suffix="_sm")
    try:
        tf.saved_model.save(model, tmp)
        return tf2onnx.convert.from_saved_model(
            tmp,
            opset=opset,
            signature_def="serving_default",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _build_viz_model(inner):
    """
    Forward pass with extra outputs after Dense / Conv2D (conv uses channel means) for the WinForms NN view.
    Final tensor is always exported as NN_output (N = sort order) so the app can find the 10-class head.
    """
    import tensorflow as tf
    from collections import OrderedDict
    from tensorflow.keras.layers import InputLayer

    inp = tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32, name="input")
    x = inp
    outputs: OrderedDict[str, tf.Tensor] = OrderedDict()
    idx = 0

    def skip_hidden_viz(layer) -> bool:
        cn = layer.__class__.__name__
        if "Random" in cn or "Augmentation" in cn:
            return True
        if isinstance(
            layer,
            (
                tf.keras.layers.Dropout,
                tf.keras.layers.Flatten,
                tf.keras.layers.MaxPooling2D,
                tf.keras.layers.AveragePooling2D,
                tf.keras.layers.BatchNormalization,
                tf.keras.layers.Add,
                tf.keras.layers.Multiply,
            ),
        ):
            return True
        return False

    layer_list = [layer for layer in inner.layers if not isinstance(layer, InputLayer)]
    for li, layer in enumerate(layer_list):
        x = layer(x)
        is_last = li == len(layer_list) - 1
        if is_last:
            outputs[f"{idx:02d}_output"] = x
            break
        if skip_hidden_viz(layer):
            continue
        safe = (layer.name or "layer").replace("/", "_")[:48]
        if isinstance(layer, tf.keras.layers.Dense):
            outputs[f"{idx:02d}_dense_{safe}"] = x
            idx += 1
        elif isinstance(layer, tf.keras.layers.Conv2D):
            outputs[f"{idx:02d}_conv_{safe}"] = tf.reduce_mean(x, axis=[1, 2])
            idx += 1
        elif isinstance(layer, tf.keras.layers.Activation):
            outputs[f"{idx:02d}_act_{safe}"] = x
            idx += 1

    return tf.keras.Model(inputs=inp, outputs=outputs, name="mnist_viz_export")


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
    viz_model = _build_viz_model(model)
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

    model_proto, _ = _keras_to_onnx(viz_model, spec, opset=13)
    with open(out_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
