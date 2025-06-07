import os, io
import numpy as np
import onnxruntime as ort
from PIL import Image

class OnnxModel:
    def __init__(self, onnx_path: str):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except:
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.mean = np.array([0.485,0.456,0.406],dtype=np.float32)
        self.std  = np.array([0.229,0.224,0.225],dtype=np.float32)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.resize((224,224), Image.BILINEAR).convert("RGB")
        arr = np.array(img).astype(np.float32)/255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2,0,1)[None,:,:,:]
        return arr

    def predict_image(self, img: Image.Image) -> int:
        inp = self.preprocess(img)
        out = self.session.run(None, {self.input_name: inp})[0]
        return int(np.argmax(out, axis=1)[0])

    def predict(self, path: str) -> int:
        from PIL import Image
        return self.predict_image(Image.open(path))
