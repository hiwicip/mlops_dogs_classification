from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from transformers import AutoImageProcessor

from dogs_classification.model import DogModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and processor")
    app.state.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    app.state.model = DogModel(model_name="google/vit-base-patch16-224")
    state_dict = torch.load("models/best_model.pt", map_location="cpu")
    app.state.model.load_state_dict(state_dict)
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
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

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
