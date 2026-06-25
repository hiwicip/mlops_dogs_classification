import json
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from evidently import Report
from evidently.presets import DataDriftPreset
from google.cloud import storage
from PIL import Image
from transformers import AutoImageProcessor

from dogs_classification.model import DogModel

BUCKET_NAME = "mlops-dog-data-euwest4"
PREDICTIONS_PREFIX = "predictions/"
REPORT_OUTPUT_PATH = "drift_reports/drift_report_{timestamp}.html"
METADATA_PATH = Path("data/processed/metadata.csv")
CLASSES_FILE = Path("data/processed/classes.json")

SAMPLES_PER_CLASS = 5

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = DogModel(model_name="google/vit-base-patch16-224")
model.eval()


def get_vit_embeddings(pixel_values: torch.Tensor):
    with torch.no_grad():
        outputs = model.model.vit(pixel_values=pixel_values)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()


def download_predictions_from_gcs(bucket_name: str, prefix: str) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    rows = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            content = blob.download_as_string()
            data = json.loads(content)
            rows.append(
                {
                    "image_path": data["image_path"],
                    "predicted_class": data["predicted_class"],
                    "confidence": data["confidence"],
                }
            )

    return pd.DataFrame(rows)


def load_reference_pixel_values(reference_df: pd.DataFrame) -> torch.Tensor:
    pixel_values = [torch.load(image_path) for image_path in reference_df["image_path"].tolist()]
    return torch.stack(pixel_values)


def load_current_pixel_values(current_df: pd.DataFrame, bucket_name: str) -> torch.Tensor:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    pixel_values = []
    for image_path in current_df["image_path"].tolist():
        image_bytes = bucket.blob(image_path).download_as_bytes()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values.append(inputs["pixel_values"].squeeze(0))

    return torch.stack(pixel_values)


def upload_report_to_gcs(local_path: str, bucket_name: str, destination_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(local_path)
    print(f"Drift report uploaded to gs://{bucket_name}/{destination_path}")


def main() -> None:
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

    reference_embeddings = get_vit_embeddings(reference_pixel_values)
    current_embeddings = get_vit_embeddings(current_pixel_values)

    reference_embed_df = pd.DataFrame(
        reference_embeddings, columns=[f"dim_{i}" for i in range(reference_embeddings.shape[1])]
    )
    reference_embed_df["label"] = reference_df["label"].values

    current_embed_df = pd.DataFrame(
        current_embeddings, columns=[f"dim_{i}" for i in range(current_embeddings.shape[1])]
    )
    current_embed_df["label"] = current_df["label"].values

    reference_embed_df["embedding_mean"] = reference_embeddings.mean(axis=1)
    current_embed_df["embedding_mean"] = current_embeddings.mean(axis=1)

    embedding_columns = [column for column in reference_embed_df.columns if column.startswith("dim_")]
    reference_embed_df = reference_embed_df[["label", "embedding_mean", *embedding_columns]]
    current_embed_df = current_embed_df[["label", "embedding_mean", *embedding_columns]]

    report = Report(metrics=[DataDriftPreset(threshold=0.7)])
    eval = report.run(reference_data=reference_embed_df, current_data=current_embed_df)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        eval.save_html(tmp.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_report_path = REPORT_OUTPUT_PATH.format(timestamp=timestamp)
        upload_report_to_gcs(tmp.name, BUCKET_NAME, gcs_report_path)


if __name__ == "__main__":
    main()
