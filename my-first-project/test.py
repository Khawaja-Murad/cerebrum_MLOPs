import os, sys
from model import OnnxModel


def main():
    if not os.path.exists("model.onnx"):
        print("ERROR: model.onnx missing")
        sys.exit(1)
    model = OnnxModel("model.onnx")
    tests = {
        "/Volumes/002-350/khawajamurad/Documents/cerebrum_MLOPs/n01440764_tench.jpeg": 0,
        "/Volumes/002-350/khawajamurad/Documents/cerebrum_MLOPs/n01667114_mud_turtle.JPEG": 35,
    }
    failed = 0
    for img, exp in tests.items():
        if not os.path.exists(img):
            print("MISSING:", img)
            failed += 1
            continue
        pred = model.predict(img)
        print(f"{img} → pred={pred}, exp={exp}")
        if pred != exp:
            print(" FAIL")
            failed += 1
    sys.exit(failed)


if __name__ == "__main__":
    main()
