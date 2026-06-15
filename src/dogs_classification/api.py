import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoImageProcessor

from dogs_classification.model import DogModel

app = FastAPI(title="Dog Breed Classification API")

MODEL_PATH = "models/best_model.pt"
MODEL_NAME = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = DogModel(model_name=MODEL_NAME)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        logits = model.model(pixel_values).logits
        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    label = model.id2label[pred.item()]

    return {
        "breed": label,
        "confidence": float(confidence.item()),
    }
