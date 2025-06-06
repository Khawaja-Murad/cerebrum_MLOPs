#!/usr/bin/env python3
"""
convert_to_onnx.py

This module provides functions to convert a PyTorch image-classification model defined
in `pytorch_model.py` (with class `Classifier`) into ONNX format, and to verify the exported
ONNX model. It is designed to be imported and used directly in a Python environment
(e.g., Google Colab) without needing any bash commands.

Usage within a Python shell or notebook (e.g., Colab):
    from convert_to_onnx import convert

    # Paths in Colab:
    weights_path = "/content/pytorch_model_weights.pth"
    onnx_path    = "/content/model.onnx"

    # Convert and verify:
    convert(weights_path, onnx_path)

You can still run it as a script from a terminal:
    python convert_to_onnx.py --weights-path /content/pytorch_model_weights.pth --onnx-path /content/model.onnx
"""

import argparse
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
import torch

# Import the user-defined PyTorch model class
# Make sure pytorch_model.py is in the same directory or in PYTHONPATH
try:
    from pytorch_model import Classifier
except ImportError:
    # Setting Classifier to None and raising an error in load_pytorch_model
    # is a good way to handle this dependency missing.
    Classifier = None


def load_pytorch_model(weights_path: str) -> torch.nn.Module:
    """
    Instantiate the `Classifier` model and load weights from `weights_path`.

    Args:
        weights_path (str): Path to the .pth file containing the model weights.

    Returns:
        model (torch.nn.Module): The loaded PyTorch model in evaluation mode.

    Raises:
        FileNotFoundError: If `weights_path` does not exist.
        RuntimeError: If loading the state dict fails.
        RuntimeError: If `Classifier` class is not available.
    """
    if Classifier is None:
        raise RuntimeError(
            "Cannot proceed: 'Classifier' class is not available. "
            "Ensure that 'pytorch_model.py' exists and defines 'Classifier'."
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at '{weights_path}'")

    # Instantiate the model
    model = Classifier()
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        # Handle potential mismatch in state dict keys if the model definition changes
        # This simple load_state_dict assumes exact match.
        # A more robust approach might use strict=False or handle key mismatches.
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Failed to load weights: {e}")

    model.eval()
    return model


def export_to_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    input_name: str = "input",
    output_name: str = "output",
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): Loaded PyTorch model (in eval mode).
        onnx_path (str): File path to save the ONNX model.
        input_name (str): Name to assign to the input node in the ONNX graph.
        output_name (str): Name to assign to the output node in the ONNX graph.
    """
    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    # Ensure the dummy input matches the expected input shape of your model
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Define dynamic axes to allow variable batch sizes
    dynamic_axes = {
        input_name: {0: "batch_size"},
        output_name: {0: "batch_size"},
    }

    print(f"[INFO] Exporting model to ONNX at '{onnx_path}'...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,  # Opset 13 is commonly supported
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )
        print(f"[INFO] Export successful.")
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


def verify_onnx(
    onnx_path: str, model: torch.nn.Module, input_name: str = "input"
) -> None:
    """
    Verify the exported ONNX model by:
      1. Running onnx.checker.check_model
      2. Running a sample inference on both PyTorch and ONNX Runtime and comparing outputs.

    Args:
        onnx_path (str): Path to the ONNX file.
        model (torch.nn.Module): The same PyTorch model that was exported (in eval mode).
        input_name (str): Name of the input node in the ONNX graph.

    Raises:
        FileNotFoundError: If `onnx_path` does not exist.
        RuntimeError: If verification fails.
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at '{onnx_path}'")

    print(f"[INFO] Loading ONNX model for verification: '{onnx_path}'")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"[INFO] ONNX model check passed.")
    except Exception as e:
        raise RuntimeError(f"ONNX model check failed: {e}")

    print(f"[INFO] Comparing PyTorch vs ONNX Runtime outputs on a random input...")
    # Generate a random input tensor, ensuring it matches the shape used for export
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        # Ensure model is on the correct device (CPU here as map_location='cpu')
        torch_out = model(dummy_input).cpu().numpy()

    # Prepare the same input for ONNX Runtime
    # Use CPUExecutionProvider unless a GPU provider is available and desired
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {input_name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)[0]

    # Compare outputs (max absolute difference)
    diff = np.max(np.abs(torch_out - ort_outs))
    print(
        f"[INFO] Max absolute difference between PyTorch and ONNX outputs: {diff:.6f}"
    )
    if diff > 1e-4:
        print(
            "[WARNING] The difference is larger than 1e-4. "
            "Check model correctness or numerical stability."
        )
    else:
        print(f"[INFO] Outputs are sufficiently close (diff <= 1e-4).")


def convert(weights_path: str, onnx_path: str) -> None:
    """
    Convert a PyTorch image-classifier to ONNX and verify it in one step.

    Args:
        weights_path (str): Path to the PyTorch model weights (.pth).
        onnx_path (str): Output path for the exported ONNX model (.onnx).

    Raises:
        Exception: If any step fails (load, export, or verify).
    """
    print(f"[INFO] Starting conversion from PyTorch to ONNX.")
    # 1. Load the PyTorch model
    model = load_pytorch_model(weights_path)

    # 2. Export to ONNX
    export_to_onnx(model, onnx_path)

    # 3. Verify the exported ONNX model
    verify_onnx(onnx_path, model)

    print(f"[INFO] Conversion complete. ONNX model saved at '{onnx_path}'.")


# Define the paths for weights and ONNX output
weights_path = "/Volumes/002-350/khawajamurad/Documents/MTailor_takehomeassignment/pytorch_model_weights.pth"
onnx_path = (
    "/Volumes/002-350/khawajamurad/Documents/MTailor_takehomeassignment/model.onnx"
)

# Call the convert function
convert(weights_path, onnx_path)
