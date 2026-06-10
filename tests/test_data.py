import pandas as pd
import torch
from dogs_classification.data import DogDataset
from torch.utils.data import Dataset


def test_dog_dataset(tmp_path):
    """Test the DogDataset class."""

    image_path = tmp_path / "image0.pt"
    torch.save(torch.zeros(3, 224, 224), image_path)

    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "image_path": [str(image_path)],
            "label": [0],
            "breed": ["breed1"],
        }
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    dataset = DogDataset(metadata_path, split="train")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1
    assert len(DogDataset(metadata_path, split="no_split")) == 0
