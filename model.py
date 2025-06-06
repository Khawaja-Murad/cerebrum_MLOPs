#!/usr/bin/env python3
"""
model.py

Defines OnnxModel for loading an ONNX image-classification model and running inference.
Uses ONNX Runtime with GPU (CUDA) if available, falling back to CPU otherwise.

Usage:
    from model import OnnxModel

    # Load the ONNX file (must exist in working directory or provide full path)
    onnx_path = "model.onnx"
    model = OnnxModel(onnx_path)

    # Run inference on a file path:
    predicted_class_id = model.predict("some_image.jpg")

    # Or run inference on a PIL.Image.Image instance:
    from PIL import Image
    img = Image.open("some_image.jpg")
    predicted_class_id = model.predict_image(img)
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image


class OnnxModel:
    """
    Loads an ONNX model and runs inference on input images.

    Attributes:
        session:            onnxruntime.InferenceSession
        input_name (str):   Name of the ONNX model's input tensor
        output_name (str):  Name of the ONNX model's output tensor
        mean (np.ndarray):  RGB mean for normalization ([0.485, 0.456, 0.406])
        std  (np.ndarray):  RGB std for normalization ([0.229, 0.224, 0.225])
    """

    def __init__(self, onnx_path: str):
        """
        Initialize the ONNX Runtime session.

        Args:
            onnx_path (str): Path to the ONNX model file (e.g., "model.onnx").
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at '{onnx_path}'")

        # Attempt to use GPU first, fallback to CPU.
        providers = []
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Grab the input and output names for easy access
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL.Image.Image into a normalized, batched numpy array.

        Steps:
          1. Convert to RGB if needed
          2. Resize to 224x224 using bilinear interpolation
          3. Convert to float32 NumPy, scale to [0, 1]
          4. Normalize per-channel by (value - mean) / std
          5. Transpose to (C, H, W) and add batch dimension

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            np.ndarray of shape (1, 3, 224, 224), dtype=float32.
        """
        # 1. Convert to RGB (in case the image is grayscale or RGBA)
        img = image.convert("RGB")

        # 2. Resize to 224x224
        img = img.resize((224, 224), Image.BILINEAR)

        # 3. To NumPy array (H, W, C), scale to [0, 1]
        arr = np.array(img).astype(np.float32) / 255.0  # shape: (224, 224, 3)

        # 4. Normalize each channel
        #    Broadcasting: arr[..., c] = (arr[..., c] - mean[c]) / std[c]
        arr = (arr - self.mean) / self.std  # still (224, 224, 3)

        # 5. Transpose to (C, H, W) and add batch dimension
        arr = arr.transpose(2, 0, 1)  # shape: (3, 224, 224)
        arr = np.expand_dims(arr, axis=0)  # shape: (1, 3, 224, 224)

        return arr

    def predict(self, image_path: str) -> int:
        """
        Load an image from disk, preprocess it, run it through the ONNX model,
        and return the predicted class ID (int).

        Args:
            image_path (str): Path to an input image file.

        Returns:
            int: The predicted class index (0–999 for ImageNet).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at '{image_path}'")

        # Load via PIL
        img = Image.open(image_path)
        return self.predict_image(img)

    def predict_image(self, image: Image.Image) -> int:
        """
        Run inference on a PIL.Image.Image and return the predicted class ID.

        Args:
            image (PIL.Image.Image): A PIL image.

        Returns:
            int: The predicted class index.
        """
        # 1. Preprocess
        input_tensor = self.preprocess(image)  # shape: (1, 3, 224, 224)

        # 2. Run ONNX Runtime
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0]  # shape: (1, 1000)

        # 3. Argmax over the class dimension
        pred_id = int(np.argmax(logits, axis=1)[0])
        return pred_id
