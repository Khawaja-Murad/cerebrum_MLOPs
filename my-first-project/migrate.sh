#!/usr/bin/env bash
set -e

echo "1️⃣  Removing function-style starter..."
rm -f main.py

echo "2️⃣  Writing FastAPI server -> app.py..."
cat > app.py << 'EOF'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from model import OnnxModel
from PIL import Image
import io

app = FastAPI(title="ONNX Inference API")

# Load ONNX model once
model = OnnxModel("model.onnx")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        class_id = model.predict_image(img)
        return JSONResponse(content={"class_id": int(class_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
EOF

echo "3️⃣  Writing inference module -> model.py..."
cat > model.py << 'EOF'
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
EOF

echo "4️⃣  Writing local test -> test.py..."
cat > test.py << 'EOF'
import os, sys
from model import OnnxModel

def main():
    if not os.path.exists("model.onnx"):
        print("ERROR: model.onnx missing"); sys.exit(1)
    model = OnnxModel("model.onnx")
    tests = {
      "test_images/n01440764_tench.JPEG": 0,
      "test_images/n01667114_mud_turtle.JPEG": 35,
    }
    failed=0
    for img,exp in tests.items():
        if not os.path.exists(img):
            print("MISSING:",img); failed+=1; continue
        pred = model.predict(img)
        print(f"{img} → pred={pred}, exp={exp}")
        if pred!=exp:
            print(" FAIL"); failed+=1
    sys.exit(failed)

if __name__=="__main__":
    main()
EOF

echo "5️⃣  Writing deployment test -> test_server.py..."
cat > test_server.py << 'EOF'
import argparse, os, requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--expected", type=int)
    args = p.parse_args()

    if not os.path.exists(args.image):
        print("Image missing:",args.image); exit(1)
    hdr = {"Authorization":f"Bearer {args.api_key}"}
    files = {"file":open(args.image,"rb")}
    r = requests.post(args.url,hdr,files=files)
    if r.status_code!=200:
        print("HTTP",r.status_code,r.text); exit(1)
    cid = r.json().get("class_id")
    print("→ class_id:",cid)
    if args.expected is not None and int(cid)!=args.expected:
        print(" MISMATCH exp",args.expected); exit(1)
    print("OK")

if __name__=="__main__":
    main()
EOF

echo "6️⃣  Writing requirements.txt..."
cat > requirements.txt << 'EOF'
fastapi
uvicorn
onnxruntime-gpu
numpy
pillow
requests
EOF

echo "7️⃣  Writing Dockerfile..."
cat > Dockerfile << 'EOF'
FROM python:3.12-bookworm

RUN apt-get update && apt-get install -y dumb-init && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8192
ENTRYPOINT ["dumb-init","--"]
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8192"]
EOF

echo "8️⃣  Overwriting cerebrium.toml..."
cat > cerebrium.toml << 'EOF'
[project]
name = "onnx-fastapi-app"
type = "custom"
runtime = "python3.12"
port = 8192

[build]
dockerfile = "Dockerfile"

[health]
path = "/health"
EOF

echo "✅ Migration complete."
echo
echo "📁 Final structure:"
ls -1

