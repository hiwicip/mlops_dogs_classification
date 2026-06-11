from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from dogs_classification.data import DogDataset


def dataset_statistics(dataset_path: str = "data/processed/metadata.csv") -> None:
    train_dataset = DogDataset(Path(dataset_path), "train")
    test_dataset = DogDataset(Path(dataset_path), "test")
    eval_dataset = DogDataset(Path(dataset_path), "eval")

    print("## Dataset Statistics\n")
    print("| Split | Images | Image Shape |")
    print("|-------|--------|-------------|")
    print(f"| Train | {len(train_dataset)} | {train_dataset[0]['pixel_values'].shape} |")
    print(f"| Test  | {len(test_dataset)} | {test_dataset[0]['pixel_values'].shape} |")
    print(f"| Eval  | {len(eval_dataset)} | {eval_dataset[0]['pixel_values'].shape} |")
    print()
    print("See workflow artifacts for label distribution plots.")

    train_label_distribution = torch.bincount(torch.tensor(train_dataset.df["label"].values))
    test_label_distribution = torch.bincount(torch.tensor(test_dataset.df["label"].values))
    eval_label_distribution = torch.bincount(torch.tensor(eval_dataset.df["label"].values))

    for title, distribution, filename in [
        ("Train", train_label_distribution, "train_label_distribution.png"),
        ("Test", test_label_distribution, "test_label_distribution.png"),
        ("Eval", eval_label_distribution, "eval_label_distribution.png"),
    ]:
        plt.bar(torch.arange(len(distribution)), distribution)
        plt.title(f"{title} label distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
