import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from dogs_classification.data import DogDataset

TEST_DIR = Path(__file__).parent

skip_if_no_data = pytest.mark.skipif(
    not os.path.exists(TEST_DIR / "metadata.csv"),
    reason="Test data files not found",
)


@skip_if_no_data
def test_dog_dataset_is_dataset_instance():
    dataset = DogDataset(TEST_DIR / "metadata.csv", split="train")
    assert isinstance(dataset, Dataset)


@skip_if_no_data
def test_dog_dataset_length():
    assert len(DogDataset(TEST_DIR / "metadata.csv", split="train")) == 1


@skip_if_no_data
def test_dog_dataset_empty_split():
    assert len(DogDataset(TEST_DIR / "metadata.csv", split="no_split")) == 0


@skip_if_no_data
def test_dog_dataset_sample_keys():
    sample = DogDataset(TEST_DIR / "metadata.csv", split="train")[0]
    assert "pixel_values" in sample
    assert "labels" in sample
    assert "breed" in sample


@skip_if_no_data
def test_dog_dataset_pixel_values_shape():
    sample = DogDataset(TEST_DIR / "metadata.csv", split="train")[0]
    assert sample["pixel_values"].shape == torch.Size([3, 224, 224])


@skip_if_no_data
def test_dog_dataset_labels_type():
    sample = DogDataset(TEST_DIR / "metadata.csv", split="train")[0]
    assert isinstance(sample["labels"], int)


@skip_if_no_data
def test_dog_dataset_breed_type():
    sample = DogDataset(TEST_DIR / "metadata.csv", split="train")[0]
    assert isinstance(sample["breed"], str)
