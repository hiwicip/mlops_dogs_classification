import json
import shutil
from pathlib import Path

import kagglehub
import pandas as pd
import torch
import typer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

RAW_DIR = Path("data/raw/Images")
PROCESSED_DIR = Path("data/processed")

TEST_SIZE = 0.1
EVAL_SIZE = 0.1
SEED = 42

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


class DogDataset(Dataset):
    """Custom dog breed dataset."""

    def __init__(self, metadata_file: Path, split: str):
        self.df = pd.read_csv(metadata_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        row = self.df.iloc[index]
        pixel_values = torch.load(row["image_path"])
        label = int(row["label"])
        breed = row["breed"]

        return {
            "pixel_values": pixel_values,
            "labels": label,
            "breed": breed,
        }


def download_data() -> Path:
    cached_path = Path(kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset"))
    source_dir = cached_path / "images" / "Images"

    if not source_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {source_dir}")

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Copying dataset to {RAW_DIR}...")
    shutil.copytree(source_dir, RAW_DIR)

    print(f"Dataset available at: {RAW_DIR}")

    return RAW_DIR


def preprocess(data_path: Path = RAW_DIR, output_folder: Path = PROCESSED_DIR) -> None:
    print("Preprocessing data...")

    data_path = download_data()

    output_folder.mkdir(parents=True, exist_ok=True)

    train_dir = output_folder / "train"
    test_dir = output_folder / "test"
    eval_dir = output_folder / "eval"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    breed_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    classes = [d.name.split("-")[-1] for d in breed_dirs]

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    with open(output_folder / "classes.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    rows = []

    train_counter = 0
    test_counter = 0
    eval_counter = 0

    for breed_dir in breed_dirs:
        breed_name = breed_dir.name.split("-")[-1]
        label = class_to_idx[breed_name]

        image_paths = list(breed_dir.glob("*.jpg"))

        if len(image_paths) == 0:
            continue

        train_images, eval_test_images = train_test_split(
            image_paths,
            test_size=TEST_SIZE + EVAL_SIZE,
            random_state=SEED,
            shuffle=True,
        )

        eval_images, test_images = train_test_split(
            eval_test_images,
            test_size=EVAL_SIZE / (TEST_SIZE + EVAL_SIZE),
            random_state=SEED,
            shuffle=True,
        )

        print(
            f"{breed_name}: " f"{len(train_images)} train / " f"{len(eval_images)} eval / " f"{len(test_images)} test"
        )

        for image_path in train_images:
            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].squeeze(0)

            save_path = train_dir / f"image_{train_counter}.pt"

            torch.save(pixel_values, save_path)

            rows.append(
                {
                    "image_path": str(save_path),
                    "label": label,
                    "breed": breed_name,
                    "split": "train",
                }
            )

            train_counter += 1

        for image_path in test_images:
            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].squeeze(0)

            save_path = test_dir / f"image_{test_counter}.pt"

            torch.save(pixel_values, save_path)

            rows.append(
                {
                    "image_path": str(save_path),
                    "label": label,
                    "breed": breed_name,
                    "split": "test",
                }
            )

            test_counter += 1

        for image_path in eval_images:
            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].squeeze(0)

            save_path = eval_dir / f"image_{eval_counter}.pt"

            torch.save(pixel_values, save_path)

            rows.append(
                {
                    "image_path": str(save_path),
                    "label": label,
                    "breed": breed_name,
                    "split": "eval",
                }
            )

            eval_counter += 1

    df = pd.DataFrame(rows)

    df.to_csv(
        output_folder / "metadata.csv",
        index=False,
    )

    print("Done!")


if __name__ == "__main__":
    typer.run(preprocess)
