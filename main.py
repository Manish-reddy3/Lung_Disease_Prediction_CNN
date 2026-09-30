from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
import numpy as np

import io
import os


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Lung Disease Prediction API",
    description="CNN-based chest X-ray classification API",
    version="1.0.0"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "lung_disease_cnn.keras"
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = (224, 224)

CLASS_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

THRESHOLD = 0.5


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Lung Disease Prediction API",
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    allowed_types = {
        "image/jpeg",
        "image/png"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG and PNG images "
                "are supported."
            )
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:

        image = Image.open(
            io.BytesIO(contents)
        )

        image = image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Resize exactly like the Colab inference pipeline
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to NumPy
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    # The saved model contains Rescaling(1/255).

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # Classification
    if probability >= THRESHOLD:

        class_id = 1

    else:

        class_id = 0

    prediction = CLASS_NAMES[
        class_id
    ]

    # Confidence
    if class_id == 1:

        confidence = probability

    else:

        confidence = 1.0 - probability

    return {
        "prediction": prediction,
        "probability": round(
            probability,
            4
        ),
        "confidence": round(
            confidence,
            4
        )
    }