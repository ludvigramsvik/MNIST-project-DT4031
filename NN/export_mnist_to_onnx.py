"""
Export the trained Keras MNIST model to ONNX for the WinForms app (MnistDrawGui).

Prerequisites (from repo root or NN folder):
  pip install tensorflow tf2onnx onnx

Usage (from NN folder, with your .keras file present):
  python export_mnist_to_onnx.py
  python export_mnist_to_onnx.py --input mnist_model_modified2_data.keras --output ../MnistDrawGui/mnist_model.onnx
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Keras MNIST model to ONNX (float NHWC).")
    parser.add_argument(
        "--input",
        default="mnist_model_modified2_data.keras",
        help="Path to the saved Keras model (.keras).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "MnistDrawGui", "mnist_model.onnx"),
        help="Output .onnx path.",
    )
    args = parser.parse_args()

    in_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    import tensorflow as tf
    import tf2onnx

    if not os.path.isfile(in_path):
        raise SystemExit(f"Model not found: {in_path}")

    model = tf.keras.models.load_model(in_path)
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(out_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
