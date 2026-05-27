from dogs_classification.data import DogDataset
from torch.utils.data import Dataset


def test_dog_dataset():
    """Test the DogDataset class."""
    dataset = DogDataset("data/raw")
    assert isinstance(dataset, Dataset)
