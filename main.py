from pathlib import Path
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "model" / "lung_disease_cnn.keras"

app = FastAPI(title="LungSight API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = keras.models.load_model(MODEL_PATH)

# --- KEEP YOUR EXISTING PREPROCESSING IF IT DIFFERS ---------------------
# Assumes: 224x224 RGB, pixels scaled to 0-1, single sigmoid output = P(PNEUMONIA).
def preprocess(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
    return np.expand_dims(np.asarray(img, dtype="float32") / 255.0, axis=0)
# ------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        x = preprocess(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    p = float(model.predict(x, verbose=0).ravel()[0])
    label = "PNEUMONIA" if p >= 0.5 else "NORMAL"
    return {"prediction": label, "probability": round(p, 4),
            "confidence": round(max(p, 1 - p), 4)}

# --- Frontend (registered last so /health, /predict and /docs keep working)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
