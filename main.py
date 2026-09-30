from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from PIL import Image, UnidentifiedImageError

import tensorflow as tf
import numpy as np

import io
import os


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="Lung Disease Prediction API",
    description="CNN-based chest X-ray classification API",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",

        # Useful if you later run frontend separately
        "http://127.0.0.1:5500",
        "http://localhost:5500",

        # Useful for common VS Code Live Server ports
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# =========================================================
# PATHS
# =========================================================

# main.py:
#
# C:\Users\saith\Lung\main.py
#
# BASE_DIR:
#
# C:\Users\saith\Lung

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "lung_disease_cnn.keras"
)


FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


INDEX_FILE = os.path.join(
    FRONTEND_DIR,
    "index.html"
)


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

THRESHOLD = 0.5


# =========================================================
# STARTUP INFORMATION
# =========================================================

print("=" * 65)
print("LUNG DISEASE PREDICTION API")
print("=" * 65)

print("\nBase directory:")
print(BASE_DIR)

print("\nModel:")
print(MODEL_PATH)

print("\nFrontend:")
print(FRONTEND_DIR)


# =========================================================
# FILE CHECKS
# =========================================================

if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not os.path.isdir(FRONTEND_DIR):

    raise FileNotFoundError(
        f"\nFrontend directory not found:\n{FRONTEND_DIR}"
    )


if not os.path.isfile(INDEX_FILE):

    raise FileNotFoundError(
        f"\nindex.html not found:\n{INDEX_FILE}"
    )


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")


# =========================================================
# MODEL INFORMATION
# =========================================================

print("\nModel input shape:")
print(model.input_shape)

print("\nModel output shape:")
print(model.output_shape)

print("\nModel summary:")

model.summary()

print("\n" + "=" * 65)


# =========================================================
# FRONTEND
# =========================================================

app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="static"
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    return FileResponse(
        INDEX_FILE
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# =========================================================
# MODEL INFO
# =========================================================

@app.get("/model-info")
async def model_info():

    return {
        "input_shape": str(
            model.input_shape
        ),

        "output_shape": str(
            model.output_shape
        ),

        "image_size": IMAGE_SIZE,

        "classes": CLASS_NAMES,

        "threshold": THRESHOLD,

        "normal": "0",

        "pneumonia": "1"
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # FILE TYPE
    # -----------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            )
        )


    # -----------------------------------------------------
    # READ FILE
    # -----------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )


    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # RESIZE
    # -----------------------------------------------------

    image = image.resize(
        IMAGE_SIZE
    )


    # -----------------------------------------------------
    # NUMPY
    # -----------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )


    # -----------------------------------------------------
    # DEBUG IMAGE VALUES
    # -----------------------------------------------------

    print("\n" + "-" * 65)
    print("NEW PREDICTION")

    print("\nImage:")
    print("Width :", image.width)
    print("Height:", image.height)
    print("Mode  :", image.mode)

    print("\nPixel statistics BEFORE model:")
    print(
        "Min :",
        float(image_array.min())
    )

    print(
        "Max :",
        float(image_array.max())
    )

    print(
        "Mean:",
        float(image_array.mean())
    )


    # -----------------------------------------------------
    # ADD BATCH DIMENSION
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    print("\nInput shape:")
    print(image_array.shape)

    print(
        "Input dtype:",
        image_array.dtype
    )


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # DO NOT:
    #
    # image_array = image_array / 255.0
    #
    # because the model already contains:
    #
    # Rescaling(1.0 / 255.0)
    #

    raw_output = model.predict(
        image_array,
        verbose=0
    )


    print("\nRaw model output:")
    print(raw_output)

    print(
        "Raw output shape:",
        raw_output.shape
    )


    # -----------------------------------------------------
    # EXTRACT PROBABILITY
    # -----------------------------------------------------

    try:

        probability = float(
            raw_output[0][0]
        )

    except (
        IndexError,
        TypeError,
        ValueError
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected model output format. "
                f"Received: {raw_output}"
            )
        )


    print(
        "\nPNEUMONIA probability:",
        probability
    )

    print(
        "NORMAL probability:",
        1.0 - probability
    )

    print(
        "Threshold:",
        THRESHOLD
    )


    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    if probability >= THRESHOLD:

        class_id = 1

    else:

        class_id = 0


    prediction = CLASS_NAMES[
        class_id
    ]


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if class_id == 1:

        confidence = probability

    else:

        confidence = 1.0 - probability


    print(
        "\nPrediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence
    )

    print("-" * 65)


    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

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