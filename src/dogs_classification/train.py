from dogs_classification.data import DogDataset


def train():
    dataset = DogDataset("data/raw", split="train")  # noqa: F841
    # add rest of your training code here. Remove noqa when model and dataset variables are used.


if __name__ == "__main__":
    train()
