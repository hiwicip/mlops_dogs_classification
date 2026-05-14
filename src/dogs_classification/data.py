import json
from pathlib import Path

import pandas as pd
import typer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

RAW_DIR = Path("data/raw/Images")
PROCESSED_DIR = Path("data/processed")

TEST_SIZE = 0.2
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
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = int(row["label"])

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }

    def preprocess(data_path: Path, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def preprocess(data_path: Path = RAW_DIR, output_folder: Path = PROCESSED_DIR) -> None:
    print("Preprocessing data...")
    # dataset = DogDataset(data_path)
    # dataset.preprocess(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    breed_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    classes = [d.name.split("-")[-1] for d in breed_dirs]

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    with open(
        output_folder / "classes.json",
        "w",
    ) as f:
        json.dump(class_to_idx, f, indent=2)

    rows = []

    for breed_dir in breed_dirs:
        breed_name = breed_dir.name.split("-")[-1]
        label = class_to_idx[breed_name]
        image_paths = []

        image_paths.extend(breed_dir.glob("*.jpg"))

        if len(image_paths) == 0:
            continue

        train_images, test_images = train_test_split(
            image_paths,
            test_size=TEST_SIZE,
            random_state=SEED,
            shuffle=True,
        )

        print(f"{breed_name}: " f"{len(train_images)} train / " f"{len(test_images)} test")

        for image_path in train_images:
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "breed": breed_name,
                    "split": "train",
                }
            )

        for image_path in test_images:
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "breed": breed_name,
                    "split": "test",
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_folder / "metadata.csv",
        index=False,
    )

    print("Done!")


if __name__ == "__main__":
    typer.run(preprocess)
