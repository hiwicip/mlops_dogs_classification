import json
from pathlib import Path

from torch import nn
from transformers import ViTForImageClassification

CLASSES_FILE = Path("data/processed/classes.json")


class DogModel(nn.Module):
    def __init__(self, model_name: str = "google/vit-base-patch16-224", classes_file: Path = CLASSES_FILE):
        super().__init__()
        with open(classes_file) as f:
            label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.forward = self.model.forward


if __name__ == "__main__":
    model = DogModel()
