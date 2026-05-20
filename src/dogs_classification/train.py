import torch

from dogs_classification.data import DogDataset
from dogs_classification.model import DogModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10):
    train_dataset = DogDataset("data/processed/metadata.csv", "train")
    test_dataset = DogDataset("data/processed/metadata.csv", "test")  # noqa: F841
    model = DogModel()
    model.to(DEVICE)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            img = batch["pixel_values"].to(DEVICE)
            target = batch["labels"].to(DEVICE)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass: compute predictions and loss
            y_pred = model(img).logits
            loss = loss_fn(y_pred, target)

            # Backward pass: compute gradients
            loss.backward()

            optimizer.step()  # Update model parameters

            statistics["train_loss"].append(loss.item())

            # Compute and record the training accuracy
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")


if __name__ == "__main__":
    train()
