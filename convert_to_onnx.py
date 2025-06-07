#!/usr/bin/env python3
"""
Convert a PyTorch image-classifier to ONNX and verify it.
"""

import os
import torch
import numpy as np
import onnx
import onnxruntime as ort

# Try to import the user's model definition
try:
    from pytorch_model import Classifier
except ImportError:
    Classifier = None


def load_pytorch_model(weights_path: str) -> torch.nn.Module:
    # Load the Classifier class and its weights from disk
    if Classifier is None:
        raise RuntimeError("'Classifier' not found. Define it in pytorch_model.py.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = Classifier()
    try:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Error loading weights: {e}")

    return model.eval()


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    input_name: str = "input",
    output_name: str = "output",
) -> None:
    # Create a dummy input and export the model to ONNX (with dynamic batch size)
    dummy = torch.randn(1, 3, 224, 224)
    dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}}

    print(f"Exporting to ONNX: {onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=13,
            export_params=True,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )
        print("Export done.")
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


def verify_onnx(
    onnx_path: str, model: torch.nn.Module, input_name: str = "input"
) -> None:
    # Check the ONNX file and compare outputs with PyTorch
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    print(f"Verifying ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Model structure OK.")

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {input_name: dummy.numpy()})[0]

    diff = np.max(np.abs(pt_out - ort_out))
    print(f"Max diff: {diff:.6f}")
    if diff > 1e-4:
        print("Warning: output difference exceeds threshold.")
    else:
        print("Outputs match within tolerance.")


def convert(weights_path: str, onnx_path: str) -> None:
    # Run the full conversion: load, export, verify
    print("Starting conversion...")
    model = load_pytorch_model(weights_path)
    export_to_onnx(model, onnx_path)
    verify_onnx(onnx_path, model)
    print(f"Done. ONNX saved at {onnx_path}")


if __name__ == "__main__":
    # Set your file paths here before running
    weights = "/Volumes/002-350/khawajamurad/Documents/MTailor_takehomeassignment/pytorch_model_weights.pth"
    onnx_out = (
        "/Volumes/002-350/khawajamurad/Documents/MTailor_takehomeassignment/model.onnx"
    )
    convert(weights, onnx_out)
