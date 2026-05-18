from transformers import ViTForImageClassification


class DogModel:
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=120, ignore_mismatched_sizes=True
        )


# num_labels=120 due to 120 dog breeds


if __name__ == "__main__":
    model = DogModel()
