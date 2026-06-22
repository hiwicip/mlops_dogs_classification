import json

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image
from transformers import AutoImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"

ONNX_MODEL_PATH = "models/dog_classifier.onnx"
GCS_BUCKET = "mlops-dog-data-euwest4"
GCS_BLOB = "models/dog_classifier.onnx"
GCP_PROJECT = "mlopsdogclassification"


def _download_model_from_gcs() -> None:
    try:
        from google.cloud import storage

        client = storage.Client(project=GCP_PROJECT)
        client.bucket(GCS_BUCKET).blob(GCS_BLOB).download_to_filename(ONNX_MODEL_PATH)
        print("Model downloaded from GCS")
    except Exception as e:
        print(f"Could not download model from GCS: {e}")


_download_model_from_gcs()


@bentoml.service
class DogBreedClassificationService:
    def __init__(self):
        super().__init__()
        # Note that the onnx must be there in order for this to work
        self.model = InferenceSession(ONNX_MODEL_PATH)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        with open("data/processed/classes.json") as f:
            label2id = json.load(f)
        self.idx_to_class = {idx: name for name, idx in label2id.items()}

    @bentoml.api
    def predict(self, image: Image.Image) -> dict:
        # Preprocess input
        image = image.convert("RGB")
        image_np = np.array(image)
        inputs = self.processor(images=image_np, return_tensors="np")

        ort_inputs = {"pixel_values": inputs["pixel_values"].astype(np.float32)}

        # Inference
        logits = self.model.run(
            None,
            ort_inputs,
        )[0]
        prediction = int(np.argmax(logits, axis=1)[0])
        # Softmax for confidence
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        confidence = float(np.max(probs))
        return {
            "predicted_class": self.idx_to_class[prediction],
            "confidence": confidence,
        }
