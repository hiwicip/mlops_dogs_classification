import json

import bentoml
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image
from transformers import AutoImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"
ONNX_MODEL_PATH = "models/dog_classifier.onnx"


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
