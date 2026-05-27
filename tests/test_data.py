import pandas as pd
from dogs_classification.data import DogDataset
from torch.utils.data import Dataset


def test_dog_dataset(tmp_path):
    """Test the DogDataset class."""

    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "image_path": ["path/to/image1.pt"],
            "label": [0],
            "breed": ["breed1"],
        }
    )

    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    dataset = DogDataset(metadata_path, split="train")
    assert isinstance(dataset, Dataset)
