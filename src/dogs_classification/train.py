from dogs_classification.data import MyDataset
from dogs_classification.model import Model


def train():
    dataset = MyDataset("data/raw")  # noqa: F841
    model = Model()  # noqa: F841
    # add rest of your training code here. Remove noqa when model and dataset variables are used.


if __name__ == "__main__":
    train()
