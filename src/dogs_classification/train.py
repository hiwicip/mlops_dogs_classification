from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from google.cloud import storage
from omegaconf import DictConfig
from tqdm import tqdm

from dogs_classification.data import DogDataset
from dogs_classification.model import DogModel
from dogs_classification.visualize import log_visualizations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUCKET_NAME = "mlops-dogs-data-euwest4"


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(cfg: DictConfig):
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    model = DogModel(model_name=cfg.model.name)
    model.to(DEVICE)

    print("Training night and day")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    wandb.init(
        project="dogs-classification",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "model_name": cfg.model.name,
        },
    )

    train_dataset = DogDataset(Path("data/processed/metadata.csv"), "train")
    test_dataset = DogDataset(Path("data/processed/metadata.csv"), "test")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

    label_to_breed = dict(zip(train_dataset.df["label"], train_dataset.df["breed"]))

    label_to_breed = dict(zip(train_dataset.df["label"], train_dataset.df["breed"]))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    statistics: dict[str, list[float]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # MODEL TRAINING
        model.train()
        preds_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as train_pbar:
            for i, batch in enumerate(train_pbar):
                img, target = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                y_pred = model(img).logits
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                accuracy = (y_pred.argmax(dim=1) == target).float().mean()

                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy.item()})

                statistics["train_loss"].append(loss.item())
                statistics["train_accuracy"].append(accuracy.item())

                preds_list.append(y_pred.detach().softmax(dim=1))
                targets_list.append(target)

                train_pbar.set_postfix(train_loss=f"{loss.item():.4f}")

                if i % 100 == 0:
                    pred_labels = y_pred.argmax(dim=1)
                    true_breeds = batch["breed"]
                    images = [
                        wandb.Image(
                            img[j].detach().cpu(),
                            caption=f"true: {true_breeds[j]} | pred: {label_to_breed[pred_labels[j].item()]}",
                        )
                        for j in range(min(3, img.shape[0]))
                    ]
                    wandb.log({"images": images})

        # MODEL EVALUATION
        model.eval()
        val_losses, val_accs = [], []
        val_preds_list, val_targets_list = [], []
        with torch.no_grad():
            with tqdm(test_dataloader, desc="Validation") as val_pbar:
                for val_batch in val_pbar:
                    img, target = val_batch["pixel_values"].to(DEVICE), val_batch["labels"].to(DEVICE)
                    y_pred = model(img).logits
                    val_losses.append(loss_fn(y_pred, target).item())
                    val_accs.append((y_pred.argmax(dim=1) == target).float().mean().item())
                    val_preds_list.extend(y_pred.argmax(dim=1).cpu().numpy())
                    val_targets_list.extend(target.cpu().numpy())
                    val_pbar.set_postfix(val_loss=f"{sum(val_losses) / len(val_losses):.4f}")

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)

        idx_to_class = label_to_breed
        log_visualizations(np.array(val_preds_list), np.array(val_targets_list), idx_to_class)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})
        statistics["val_loss"].append(val_loss)
        statistics["val_accuracy"].append(val_acc)
        tqdm.write(f"Epoch {epoch} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tqdm.write(f"Saved best model (val_loss={val_loss:.4f})")

    # Upload results .pt file to GCS bucket

    client = storage.Client(project="mlopsdogclassification")
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("models/best_model.pt")
    blob.upload_from_filename(output_dir / "best_model.pt")

    print("Uploaded results to Bucket.")


if __name__ == "__main__":
    train()
