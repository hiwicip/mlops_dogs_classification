from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from google.cloud import storage

if TYPE_CHECKING:
    import torch
    from onnxruntime import InferenceSession
    from transformers import AutoImageProcessor

BUCKET_NAME = "mlops-dog-data-euwest4"
PREDICTIONS_PREFIX = "predictions/"
REPORT_LOCAL_PATH = "monitoring.html"
REPORT_OUTPUT_PATH = "drift_reports/drift_report_{timestamp}.html"
METADATA_PATH = Path("data/processed/metadata.csv")
CLASSES_FILE = Path("data/processed/classes.json")
ONNX_MODEL_GCS_PATH = Path("models/dog_classifier.onnx")
PROCESSED_DVC_FILE = Path("data/processed.dvc")
PROCESSED_DIR = Path("data/processed")
DOWNLOAD_WORKERS = 8
REFERENCE_REPORT_CACHE_PREFIX = "drift_reference/"

SAMPLES_PER_CLASS = 5
FEATURE_NAMES = ["brightness", "contrast", "sharpness"]

processor: AutoImageProcessor | None = None
model: InferenceSession | None = None


def download_onnx_from_gcs() -> None:
    """Download the ONNX model and its external-data sidecar from GCS to the local path."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    ONNX_MODEL_GCS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Downloads both dog_classifier.onnx and dog_classifier.onnx.data
    for blob in bucket.list_blobs(prefix=str(ONNX_MODEL_GCS_PATH)):
        destination = ONNX_MODEL_GCS_PATH.parent / Path(blob.name).name
        blob.download_to_filename(str(destination))
        print(f"Downloaded gs://{BUCKET_NAME}/{blob.name} -> {destination}")


def load_model_and_processor() -> None:
    """Load the ONNX model and image processor into module globals if not already loaded."""
    from onnxruntime import InferenceSession
    from transformers import AutoImageProcessor

    global processor, model
    if processor is None:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    if model is None:
        if not ONNX_MODEL_GCS_PATH.exists():
            download_onnx_from_gcs()
        model = InferenceSession(str(ONNX_MODEL_GCS_PATH))


# def get_vit_embeddings(pixel_values: torch.Tensor):
#     with torch.no_grad():
#         outputs = model.model.vit(pixel_values=pixel_values)
#     embeddings = outputs.last_hidden_state[:, 0, :]
#     return embeddings.cpu().numpy()


def download_predictions_from_gcs(bucket_name: str, prefix: str, n: int | None = None) -> pd.DataFrame:
    """Download prediction JSON files from GCS and return them as a DataFrame.

    Args:
        bucket_name: The name of the GCS bucket.
        prefix: The prefix for the prediction files in the GCS bucket.
        n: If given, only the latest n prediction files are used.

    Returns:
        A DataFrame with image path, predicted class, and confidence.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".json")]

    if n is not None:
        blobs = sorted(blobs, key=lambda b: b.updated, reverse=True)[:n]

    rows = []
    for blob in blobs:
        content = blob.download_as_string()
        data = json.loads(content)
        image_path = data["image_path"]
        if not bucket.blob(image_path).exists(client):
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


@lru_cache(maxsize=1)
def _dvc_relpath_to_md5() -> dict[str, str]:
    """Map each file tracked under data/processed/ to its DVC content hash."""
    with open(PROCESSED_DVC_FILE) as f:
        out = yaml.safe_load(f)["outs"][0]
    return {entry["relpath"]: entry["md5"] for entry in out["files"]}


def _download_dvc_tracked_file(bucket: storage.Bucket, image_path: str, hashes: dict[str, str]) -> bytes:
    """Fetch a single DVC-tracked file straight from the remote's content-addressed cache."""
    relpath = str(Path(image_path).relative_to(PROCESSED_DIR))
    md5 = hashes[relpath]
    blob_path = f"files/md5/{md5[:2]}/{md5[2:]}"
    return bucket.blob(blob_path).download_as_bytes()


def load_reference_pixel_values(reference_df: pd.DataFrame) -> torch.Tensor:
    """Load reference pixel-value tensors, from local disk if present, else from the DVC remote cache."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    hashes = _dvc_relpath_to_md5()

    def _load_one(image_path: str) -> torch.Tensor:
        local = Path(image_path)
        if local.exists():
            return torch.load(local)
        data = _download_dvc_tracked_file(bucket, image_path, hashes)
        return torch.load(BytesIO(data))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        pixel_values = list(executor.map(_load_one, reference_df["image_path"].tolist()))
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
    import torch
    from PIL import Image

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    assert processor is not None, "Call load_model_and_processor() before loading pixel values"

    def _load_one(image_path: str) -> torch.Tensor:
        image_bytes = bucket.blob(image_path).download_as_bytes()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")  # type: ignore[operator]
        return inputs["pixel_values"].squeeze(0)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        pixel_values = list(executor.map(_load_one, current_df["image_path"].tolist()))

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


def compute_confidence(model: InferenceSession, pixel_values: np.ndarray) -> np.ndarray:
    """
    Compute the confidence scores for the predictions made by the model on the given pixel values.

    Args:
        model (InferenceSession): The ONNX model used for making predictions.
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


def build_reference_df() -> pd.DataFrame:
    """Build the reference dataframe from the processed metadata (train split, sampled per class)."""
    reference_df = pd.read_csv(METADATA_PATH)
    reference_df = reference_df[reference_df["split"] == "train"].reset_index(drop=True)
    reference_df = (
        reference_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    return reference_df


def build_reference_report_df() -> pd.DataFrame:
    """Compute the reference-side report columns (features + confidence) from scratch.

    This is the CPU-heavy part (image loading + ONNX inference over the whole reference
    sample), so callers should go through load_or_build_reference_report_df() to avoid
    redoing it on every cold start.
    """
    load_model_and_processor()

    with open(CLASSES_FILE) as f:
        class_to_idx = json.load(f)

    reference_df = build_reference_df()
    reference_df["label"] = reference_df["breed"].map(class_to_idx).astype(int)

    reference_pixel_values = load_reference_pixel_values(reference_df)
    reference_df[FEATURE_NAMES] = extract_features(reference_pixel_values)
    reference_df["confidence"] = compute_confidence(model, reference_pixel_values.numpy())

    reference_df = reference_df[["label", "confidence", *FEATURE_NAMES]]
    reference_df["label"] = reference_df["label"].astype("category")
    return reference_df


def _reference_report_cache_blob_name() -> str:
    """Content-hash-based cache key so the cache self-invalidates when the reference data changes."""
    digest = hashlib.md5(METADATA_PATH.read_bytes()).hexdigest()
    return f"{REFERENCE_REPORT_CACHE_PREFIX}reference_report_{digest}.parquet"


def load_or_build_reference_report_df() -> pd.DataFrame:
    """Load the cached reference report from GCS, computing and caching it if missing."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(_reference_report_cache_blob_name())

    if blob.exists(client):
        print(f"Loading cached reference report from gs://{BUCKET_NAME}/{blob.name}")
        return pd.read_parquet(BytesIO(blob.download_as_bytes()))

    print("No cached reference report found, computing it now")
    reference_report_df = build_reference_report_df()

    buffer = BytesIO()
    reference_report_df.to_parquet(buffer, index=False)
    buffer.seek(0)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    print(f"Cached reference report to gs://{BUCKET_NAME}/{blob.name}")

    return reference_report_df


def run_analysis(reference_report_df: pd.DataFrame, current_df: pd.DataFrame) -> str:
    """
    Run the drift analysis between the (precomputed) reference report data and current data.

    Returns:
        str: The local path to the saved HTML report.
    """
    load_model_and_processor()

    with open(CLASSES_FILE) as f:
        class_to_idx = json.load(f)

    current_df["label"] = current_df["predicted_class"].map(class_to_idx)
    current_df = current_df.dropna(subset=["label"]).copy()
    current_df["label"] = current_df["label"].astype(int)

    current_pixel_values = load_current_pixel_values(current_df, BUCKET_NAME)
    current_df[FEATURE_NAMES] = extract_features(current_pixel_values)

    current_df = current_df[["label", "confidence", *FEATURE_NAMES]]
    current_df["label"] = current_df["label"].astype("category")

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=reference_report_df, current_data=current_df)
    result.save_html(REPORT_LOCAL_PATH)
    return REPORT_LOCAL_PATH


def main() -> None:
    """Generate a drift report from the latest predictions and upload it to GCS."""
    current_df = download_predictions_from_gcs(BUCKET_NAME, PREDICTIONS_PREFIX)
    if current_df.empty:
        print("No predictions found in the bucket.")
        return

    reference_report_df = load_or_build_reference_report_df()
    local_path = run_analysis(reference_report_df, current_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_report_to_gcs(local_path, BUCKET_NAME, REPORT_OUTPUT_PATH.format(timestamp=timestamp))


if __name__ == "__main__":
    main()
