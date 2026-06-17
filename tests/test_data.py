import os
from pathlib import Path

import pytest
import torch
from dogs_classification.data import DogDataset
from torch.utils.data import Dataset

TEST_DIR = Path(__file__).parent


@pytest.mark.skipif(
    not os.path.exists(TEST_DIR / "metadata.csv"),
    reason="Test data files not found",
)
def test_dog_dataset():
    dataset = DogDataset(TEST_DIR / "metadata.csv", split="train")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1
    assert len(DogDataset(TEST_DIR / "metadata.csv", split="no_split")) == 0

    sample = dataset[0]
    assert "pixel_values" in sample
    assert "labels" in sample
    assert "breed" in sample

    assert sample["pixel_values"].shape == torch.Size([3, 224, 224])
    assert isinstance(sample["labels"], int)
    assert isinstance(sample["breed"], str)
