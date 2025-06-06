#!/usr/bin/env python3
"""
test.py

Verifies that the ONNX model (model.onnx) and the inference code (model.py) work correctly.
For each sample image, it runs the model and compares the predicted class ID against
an expected class ID. If any mismatch or missing file occurs, the script exits with code 1.
If all tests pass, it exits with code 0.

Usage (from the repo root, where model.onnx lives):

    python test.py

Requirements:
    - model.onnx must exist in the current directory.
    - test images must be placed under "test_images/" (relative path).
    - The mapping `test_images → expected_class_id` must be updated to match your actual files.
"""

import os
import sys
from model import OnnxModel


def main():
    # 1. Check that the ONNX file exists
    onnx_path = "model.onnx"
    if not os.path.exists(onnx_path):
        print(
            f"[ERROR] ONNX model not found at '{onnx_path}'. Please generate it first."
        )
        sys.exit(1)

    # 2. Instantiate the inference model
    try:
        model = OnnxModel(onnx_path)
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        sys.exit(1)

    # 3. Define test images and their expected class IDs.
    #
    #    You must place these image files under a folder named "test_images/" in your repo root.
    #    For example:
    #      - test_images/n01440764_tench.JPEG
    #      - test_images/n01667114_mud_turtle.JPEG
    #
    #    The numbers below (0 and 35) are the ImageNet class IDs for "tench" and "mud turtle".
    #    Adjust paths and IDs to match your actual test images.
    test_images = {
        "n01440764_tench.jpeg": 0,
        "n01667114_mud_turtle.JPEG": 35,
    }

    all_passed = True

    for img_path, expected_id in test_images.items():
        # 3a. Check that the test image exists
        if not os.path.exists(img_path):
            print(f"[ERROR] Test image not found: '{img_path}'")
            all_passed = False
            continue

        # 3b. Run inference
        try:
            predicted_id = model.predict(img_path)
        except Exception as e:
            print(f"[ERROR] Inference failed on '{img_path}': {e}")
            all_passed = False
            continue

        # 3c. Compare predicted vs expected
        print(
            f"[TEST] {img_path}  →  Predicted: {predicted_id}, Expected: {expected_id}"
        )
        if predicted_id != expected_id:
            print(
                f"[FAIL] Mismatch for '{img_path}': "
                f"predicted {predicted_id}, expected {expected_id}"
            )
            all_passed = False

    # 4. Final exit code
    if all_passed:
        print("[INFO] All tests passed.")
        sys.exit(0)
    else:
        print("[INFO] Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
