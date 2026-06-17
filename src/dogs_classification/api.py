import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import AutoImageProcessor

from dogs_classification.model import DogModel

MODEL_PATH = "models/best_model.pt"
GCS_BUCKET = "mlops-dog-data-euwest4"
GCS_BLOB = "models/best_model.pt"
GCP_PROJECT = "mlopsdogclassification"


def _download_model_from_gcs() -> None:
    try:
        from google.cloud import storage
        client = storage.Client(project=GCP_PROJECT)
        client.bucket(GCS_BUCKET).blob(GCS_BLOB).download_to_filename(MODEL_PATH)
        print("Model downloaded from GCS")
    except Exception as e:
        print(f"Could not download model from GCS: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and processor")
    if not os.path.exists(MODEL_PATH):
        _download_model_from_gcs()
    app.state.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    app.state.model = DogModel(model_name="google/vit-base-patch16-224")
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        app.state.model.load_state_dict(state_dict)
    else:
        print(f"No fine-tuned weights found at {MODEL_PATH}, using pretrained model")
    app.state.model.eval()

    yield

    print("Cleaning up resources")
    del app.state.model, app.state.processor


app = FastAPI(title="Dog Breed Classification API", lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Welcome to the Dogs Classification API!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid image file") from err

    inputs = app.state.processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        logits = app.state.model.model(pixel_values).logits
        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    label = app.state.model.id2label[pred.item()]

    return {
        "breed": label,
        "confidence": float(confidence.item()),
    }
