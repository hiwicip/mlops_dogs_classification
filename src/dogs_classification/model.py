import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from transformers import ViTForImageClassification

from dogs_classification.data import DogDataset
from dogs_classification.visualize import log_visualizations

CLASSES_FILE = Path("data/processed/classes.json")


class DogModel(LightningModule):
    """
    A PyTorch Lightning module for dog breed classification using a pre-trained Vision Transformer (ViT) model.
    This class handles the training, validation, and testing of the model, as well as logging
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        classes_file: Path = CLASSES_FILE,
        lr: float = 0.001,
        batch_size: int = 32,
        test_out_dir: str = "logs/eval",
    ):
        super().__init__()
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if batch_size < 1:
            raise ValueError(f"Batch size must be at least 1, got {batch_size}")
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
        self.batch_size = batch_size
        self.id2label = id2label
        self.criterium = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.val_outputs: list[dict[str, np.ndarray]] = []
        self.test_outputs: list[dict[str, np.ndarray]] = []
        self.test_out_dir = test_out_dir

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.
        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        img, target = batch["pixel_values"].to(self.device), batch["labels"].to(self.device)
        if img.ndim != 4 or img.shape[1] != 3 or img.shape[2] != 224 or img.shape[3] != 224:
            raise ValueError(f"Expected input shape [batch, 3, 224, 224], got {list(img.shape)}")
        y_pred = self.model(img).logits
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=True, prog_bar=True)
        if batch_idx % 100 == 0 and self.logger is not None:
            pred_labels = y_pred.argmax(dim=1)
            true_breeds = batch["breed"]
            self.logger.experiment.log(
                {
                    "images": [
                        wandb.Image(
                            img[j].detach().cpu(),
                            caption=f"true: {true_breeds[j]} | pred: {self.id2label[pred_labels[j].item()]}",
                        )
                        for j in range(min(3, img.shape[0]))
                    ]
                }
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Perform a single validation step.
        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.
        Returns:
            None
        """
        img, target = batch["pixel_values"].to(self.device), batch["labels"].to(self.device)
        y_pred = self.model(img).logits
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        self.val_outputs.append(
            {
                "preds": y_pred.argmax(dim=1).cpu().numpy(),
                "targets": target.cpu().numpy(),
            }
        )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        Perform a single test step.
        Args:
            batch (dict): A batch of data containing 'pixel_values' and 'labels'.
            batch_idx (int): The index of the batch.
        Returns:
            None
        """
        img, target = batch["pixel_values"].to(self.device), batch["labels"].to(self.device)
        y_pred = self.model(img).logits
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log("test_accuracy", acc, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        self.test_outputs.append(
            {
                "preds": y_pred.argmax(dim=1).cpu().numpy(),
                "targets": target.cpu().numpy(),
            }
        )

    def on_validation_epoch_end(self):
        """Summarize validation results at the end of the epoch and log visualizations."""
        all_preds = np.concatenate([o["preds"] for o in self.val_outputs])
        all_targets = np.concatenate([o["targets"] for o in self.val_outputs])
        if self.logger is not None:
            log_visualizations(all_preds, all_targets, self.id2label, self.logger.experiment)
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        """Summarize test results at the end of the epoch and log visualizations."""
        all_preds = np.concatenate([o["preds"] for o in self.test_outputs])
        all_targets = np.concatenate([o["targets"] for o in self.test_outputs])
        if self.logger is not None:
            log_visualizations(all_preds, all_targets, self.id2label, self.logger.experiment)
        self.test_outputs.clear()

        correct_per_class: dict[int, int] = defaultdict(int)
        total_per_class: dict[int, int] = defaultdict(int)

        for pred, label in zip(all_preds, all_targets, strict=True):
            total_per_class[label] += 1
            if pred == label:
                correct_per_class[label] += 1

        class_accuracy = {int(cls): correct_per_class[cls] / total_per_class[cls] for cls in total_per_class}

        if self.logger is not None:
            table = wandb.Table(columns=["class", "accuracy"])
            for cls, acc in class_accuracy.items():
                table.add_data(cls, acc)
            self.logger.experiment.log({"class_accuracy": table})

        output_dir = Path(self.test_out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "all_preds.npy", np.array(all_preds))
        np.save(output_dir / "all_labels.npy", np.array(all_targets))

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        """Create the training data loader."""
        train_dataset = DogDataset(Path("data/processed/metadata.csv"), "train")
        return torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Create the validation data loader."""
        val_dataset = DogDataset(Path("data/processed/metadata.csv"), "test")
        return torch.utils.data.DataLoader(val_dataset, self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Create the test data loader."""
        test_dataset = DogDataset(Path("data/processed/metadata.csv"), "eval")
        return torch.utils.data.DataLoader(test_dataset, self.batch_size, shuffle=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            pixel_values (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224].
        Returns:
            torch.Tensor: The logits output from the model.
        """
        return self.model(pixel_values).logits

    def evaluate(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                logits = self(batch["pixel_values"].to(self.device))
                preds = logits.argmax(dim=1)
                labels = batch["labels"].to(self.device)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    model = DogModel()
