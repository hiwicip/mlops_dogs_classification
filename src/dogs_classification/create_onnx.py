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
gcs_checkpoint_prefix = "models"

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

checkpoint_path.mkdir(parents=True, exist_ok=True)

for blob in bucket.list_blobs(prefix=gcs_checkpoint_prefix):
    if blob.name.endswith("/"):
        continue

    rel_path = Path(blob.name).relative_to(gcs_checkpoint_prefix)
    dest_path = checkpoint_path / rel_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dest_path)

    print(f"Downloaded {blob.name} -> {dest_path}")

checkpoint_file = next(checkpoint_path.glob("best_model_*.pt"))
print(f"Using checkpoint: {checkpoint_file}")

model = DogModel(model_name=MODEL_NAME)
state_dict = torch.load(checkpoint_file, map_location="cpu")
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
print(onnx.helper.printable_graph(onnx_model.graph))
