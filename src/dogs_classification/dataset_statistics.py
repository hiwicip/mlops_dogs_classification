from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from utils import show_image_and_target

from dogs_classification.data import DogDataset


def dataset_statistics(dataset_path: str = "data/processed/metadata.csv") -> None:
    train_dataset = DogDataset(Path(dataset_path), "train")
    test_dataset = DogDataset(Path(dataset_path), "test")
    print("Train dataset:")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0]['pixel_values'].shape}")
    print("\n")
    print("Test dataset:")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0]['pixel_values'].shape}")

    show_image_and_target(train_dataset.images[:5], train_dataset.target[:5])
    plt.savefig("dog_images.png")
    plt.close()

    train_label_distribution = torch.bincount(torch.tensor(train_dataset.df["label"].values))
    test_label_distribution = torch.bincount(torch.tensor(test_dataset.df["label"].values))

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
