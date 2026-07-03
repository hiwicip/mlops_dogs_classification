import json
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from google.cloud import storage
from onnxruntime import InferenceSession
from PIL import Image
from transformers import AutoImageProcessor

BUCKET_NAME = "mlops-dog-data-euwest4"
PREDICTIONS_PREFIX = "predictions/"
REPORT_OUTPUT_PATH = "drift_reports/drift_report_{timestamp}.html"
METADATA_PATH = Path("data/processed/metadata.csv")
CLASSES_FILE = Path("data/processed/classes.json")
ONNX_MODEL_PATH = Path("models/dog_classifier.onnx")

SAMPLES_PER_CLASS = 5

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = InferenceSession(ONNX_MODEL_PATH)


# def get_vit_embeddings(pixel_values: torch.Tensor):
#     with torch.no_grad():
#         outputs = model.model.vit(pixel_values=pixel_values)
#     embeddings = outputs.last_hidden_state[:, 0, :]
#     return embeddings.cpu().numpy()


def download_predictions_from_gcs(bucket_name: str, prefix: str) -> pd.DataFrame:
    """
    Download prediction JSON files from Google Cloud Storage (GCS) and return a DataFrame containing the predictions.
    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix for the prediction files in the GCS bucket.
    Returns:
        pd.DataFrame: A DataFrame containing the predictions with columns for image path, predicted class,
        and confidence.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    rows = []

    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue

        content = blob.download_as_string()
        data = json.loads(content)

        image_path = data["image_path"]

        # Prüfen, ob das referenzierte Bild existiert
        image_blob = bucket.blob(image_path)
        if not image_blob.exists(client):
            print(f"Skipping prediction because image is missing: {image_path}")
            continue

        rows.append(
            {
                "image_path": image_path,
                "predicted_class": data["predicted_class"],
                "confidence": data["confidence"],
            }
        )

    return pd.DataFrame(rows)


def load_reference_pixel_values(reference_df: pd.DataFrame) -> torch.Tensor:
    """
    Load pixel values for the reference dataset from the local file system.
    Args:
        reference_df (pd.DataFrame): A DataFrame containing the reference dataset with image paths.
    Returns:
        torch.Tensor: A tensor containing the pixel values of the reference images.
    """
    pixel_values = [torch.load(image_path) for image_path in reference_df["image_path"].tolist()]
    return torch.stack(pixel_values)


def load_current_pixel_values(current_df: pd.DataFrame, bucket_name: str) -> torch.Tensor:
    """
    Load pixel values for the current dataset from Google Cloud Storage (GCS).
    Args:
        current_df (pd.DataFrame): A DataFrame containing the current dataset with image paths.
        bucket_name (str): The name of the GCS bucket.
    Returns:
        torch.Tensor: A tensor containing the pixel values of the current images.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    pixel_values = []
    for image_path in current_df["image_path"].tolist():
        image_bytes = bucket.blob(image_path).download_as_bytes()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values.append(inputs["pixel_values"].squeeze(0))

    return torch.stack(pixel_values)


def extract_features(pixel_values: torch.Tensor) -> np.ndarray:
    """
    Extract features (brightness, contrast, sharpness) from the pixel values of images.
    Args:
        pixel_values (torch.Tensor): A tensor containing the pixel values of images.
    Returns:
        np.ndarray: A NumPy array containing the extracted features for each image.
    """
    features = []

    for img in pixel_values.numpy():
        brightness = float(img.mean())
        contrast = float(img.std())

        gx, gy = np.gradient(img, axis=(-2, -1))

        sharpness = float(np.mean(np.sqrt(gx**2 + gy**2)))

        features.append([brightness, contrast, sharpness])

    return np.array(features)


def compute_confidence(model, pixel_values: np.ndarray) -> np.ndarray:
    """
    Compute the confidence scores for the predictions made by the model on the given pixel values.
    Args:
        model: The ONNX model used for making predictions.
        pixel_values (np.ndarray): A NumPy array containing the pixel values of images.
    Returns:
        np.ndarray: A NumPy array containing the confidence scores for each image.
    """
    confidences = []

    for x in pixel_values:
        x = x[None, ...].astype(np.float32)  # (1, C, H, W)

        logits = model.run(None, {"pixel_values": x})[0]

        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        confidences.append(float(np.max(probs)))

    return np.array(confidences)


def upload_report_to_gcs(local_path: str, bucket_name: str, destination_path: str) -> None:
    """
    Upload the generated drift report to Google Cloud Storage (GCS).
    Args:
        local_path (str): The local path to the drift report file.
        bucket_name (str): The name of the GCS bucket.
        destination_path (str): The destination path in the GCS bucket where the report will be uploaded.
    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    print(f"Drift report uploaded to gs://{bucket_name}/{destination_path}")


def main() -> None:
    """
    Main function to perform data drift analysis between the reference dataset and the current predictions.
    It downloads the predictions from GCS, loads the reference and current pixel values, extracts features,
    computes confidence scores, and generates a drift report which is then uploaded to GCS.
    Args:
        None
    Returns:
        None
    """
    reference_df = pd.read_csv(METADATA_PATH)
    reference_df = reference_df[reference_df["split"] == "train"].reset_index(drop=True)
    reference_df = (
        reference_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )

    current_df = download_predictions_from_gcs(BUCKET_NAME, PREDICTIONS_PREFIX)

    if current_df.empty:
        print("No predictions found in the bucket.")
        return

    with open(CLASSES_FILE) as f:
        class_to_idx = json.load(f)

    current_df["label"] = current_df["predicted_class"].map(class_to_idx)
    current_df = current_df.dropna(subset=["label"]).copy()
    current_df["label"] = current_df["label"].astype(int)

    reference_df["label"] = reference_df["breed"].map(class_to_idx)
    reference_df["label"] = reference_df["label"].astype(int)

    if current_df.empty:
        print("No usable predictions found in the bucket.")
        return

    reference_pixel_values = load_reference_pixel_values(reference_df)
    current_pixel_values = load_current_pixel_values(current_df, BUCKET_NAME)

    reference_features = extract_features(reference_pixel_values)
    current_features = extract_features(current_pixel_values)

    feature_names = ["brightness", "contrast", "sharpness"]

    reference_df[feature_names] = reference_features
    current_df[feature_names] = current_features

    # Compute confidence for reference data
    reference_confidence = compute_confidence(model, reference_pixel_values.numpy())
    reference_df["confidence"] = reference_confidence

    # only keep the relevant columns for the drift report
    reference_df = reference_df[["label", "confidence", *feature_names]]
    current_df = current_df[["label", "confidence", *feature_names]]

    # change label column to categorical for the drift report
    reference_df["label"] = reference_df["label"].astype("category")
    current_df["label"] = current_df["label"].astype("category")

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    eval = report.run(reference_data=reference_df, current_data=current_df)

    # reference_embeddings = get_vit_embeddings(reference_pixel_values)
    # current_embeddings = get_vit_embeddings(current_pixel_values)

    # reference_embed_df = pd.DataFrame(
    #     reference_embeddings, columns=[f"dim_{i}" for i in range(reference_embeddings.shape[1])]
    # )
    # reference_embed_df["label"] = reference_df["label"].values

    # current_embed_df = pd.DataFrame(
    #     current_embeddings, columns=[f"dim_{i}" for i in range(current_embeddings.shape[1])]
    # )
    # current_embed_df["label"] = current_df["label"].values

    # reference_embed_df["embedding_mean"] = reference_embeddings.mean(axis=1)
    # current_embed_df["embedding_mean"] = current_embeddings.mean(axis=1)

    # embedding_columns = [column for column in reference_embed_df.columns if column.startswith("dim_")]
    # reference_embed_df = reference_embed_df[["label", "embedding_mean", *embedding_columns]]
    # current_embed_df = current_embed_df[["label", "embedding_mean", *embedding_columns]]

    # report = Report(metrics=[DataDriftPreset(threshold=0.7)])
    # eval = report.run(reference_data=reference_embed_df, current_data=current_embed_df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        eval.save_html(tmp.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_report_path = REPORT_OUTPUT_PATH.format(timestamp=timestamp)
        upload_report_to_gcs(tmp.name, BUCKET_NAME, gcs_report_path)


if __name__ == "__main__":
    main()
