import json
from datetime import UTC, datetime
from io import BytesIO
from threading import Thread

import bentoml
import numpy as np
from google.cloud import storage
from onnxruntime import InferenceSession
from PIL import Image
from transformers import AutoImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"
ONNX_MODEL_PATH = "models/dog_classifier.onnx"
BUCKET_NAME = "mlops-dog-data-euwest4"
PROJECT_ID = "mlopsdogclassification"


def save_prediction(timestamp: str, image: Image.Image, predicted_class: str, confidence: float, predictions: list):
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)

        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        image_filename = f"input_images/{timestamp}.jpg"

        bucket.blob(image_filename).upload_from_file(image_buffer, content_type="image/jpeg")

        data = {
            "timestamp": timestamp,
            # Top 1 (für drift report)
            "predicted_class": predicted_class,
            "confidence": confidence,
            # Top 5
            "predictions": predictions,
            "image_path": image_filename,
        }

        filename = f"predictions/prediction_{timestamp}.json"
        blob = bucket.blob(filename)
        blob.upload_from_string(json.dumps(data))

        print(f"Prediction saved to GCP bucket: {filename}")

    except Exception as e:
        print(f"Failed to save prediction: {e}")


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
        logits = self.model.run(None, ort_inputs)[0]
        prediction = int(np.argmax(logits, axis=1)[0])

        # Softmax for confidence
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        confidence = float(np.max(probs))

        predicted_class = self.idx_to_class[prediction]

        # Top 5 predictions
        probs = probs[0]
        top5_indices = np.argsort(probs)[::-1][:5]
        predictions = {f"class{i + 1}": self.idx_to_class[int(idx)] for i, idx in enumerate(top5_indices)}

        predictions.update({f"confidence{i + 1}": float(probs[idx]) for i, idx in enumerate(top5_indices)})

        timestamp = datetime.now(UTC).isoformat()

        # Save prediction and image to GCP bucket in a separate thread
        Thread(
            target=save_prediction,
            args=(timestamp, image.copy(), predicted_class, confidence, predictions),
            daemon=True,
        ).start()

        return {
            "predictions": predictions,
        }
