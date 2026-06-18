from pathlib import Path

import onnx
import torch
from google.cloud import storage
from PIL import Image
from transformers import AutoImageProcessor

from dogs_classification.model import DogModel

BUCKET_NAME = "mlops-dog-data-euwest4"
PROJECT_ID = "mlopsdogclassification"

MODEL_NAME = "google/vit-base-patch16-224"

checkpoint_path = Path("models")

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

checkpoint_path.mkdir(parents=True, exist_ok=True)

blob = bucket.blob("models/best_model.pt")

local_path = checkpoint_path / "best_model.pt"
blob.download_to_filename(local_path)

print(f"Downloaded {blob.name} -> {local_path}")

model = DogModel(model_name=MODEL_NAME)
state_dict = torch.load(local_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

dummy_image = Image.new("RGB", (224, 224), color="red")

inputs = processor(
    images=dummy_image,
    return_tensors="pt",
)

onnx_path = checkpoint_path / "dog_classifier.onnx"

torch.onnx.export(
    model,
    inputs["pixel_values"],
    str(onnx_path),
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
)

print(f"Model exported to {onnx_path}")

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

print("ONNX model verified.")
