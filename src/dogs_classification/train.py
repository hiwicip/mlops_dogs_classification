import contextlib
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from tqdm import tqdm

from dogs_classification.data import DogDataset
from dogs_classification.model import DogModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(cfg: DictConfig):
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    model = DogModel(model_name=cfg.model.name)
    model.to(DEVICE)
    train_dataset = DogDataset(Path("data/processed/metadata.csv"), "train")
    test_dataset = DogDataset(Path("data/processed/metadata.csv"), "test")  # noqa: F841
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)  # noqa: F841

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    statistics: dict[str, list[float]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler("./logs/profiler"),
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    ) if cfg.training.profile else contextlib.nullcontext() as prof:
        for epoch in range(epochs):
            # MODEL TRAINING
            model.train()

            for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
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

                if cfg.training.profile and prof is not None:
                    prof.step()

            # MODEL EVALUATION
            model.eval()
            val_losses, val_accs = [], []
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Validation"):
                    img = batch["pixel_values"].to(DEVICE)
                    target = batch["labels"].to(DEVICE)
                    y_pred = model(img).logits
                    val_losses.append(loss_fn(y_pred, target).item())
                    val_accs.append((y_pred.argmax(dim=1) == target).float().mean().item())

            val_loss = sum(val_losses) / len(val_losses)
            val_acc = sum(val_accs) / len(val_accs)

            statistics["val_loss"].append(val_loss)
            statistics["val_accuracy"].append(val_acc)
            tqdm.write(f"Epoch {epoch} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                tqdm.write(f"Saved best model (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    train()
