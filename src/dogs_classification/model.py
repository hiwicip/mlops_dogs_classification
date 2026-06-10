import json
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
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        classes_file: Path = CLASSES_FILE,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
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
        self.batch_size = batch_size
        self.id2label = id2label
        self.criterium = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.val_outputs: list[dict[str, np.ndarray]] = []

    def training_step(self, batch, batch_idx):
        img, target = batch["pixel_values"].to(self.device), batch["labels"].to(self.device)
        y_pred = self.model(img).logits
        loss = self.criterium(y_pred, target)
        acc = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=True, prog_bar=True)
        if batch_idx % 100 == 0:
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

    def validation_step(self, batch, batch_idx):
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

    def on_validation_epoch_end(self):
        all_preds = np.concatenate([o["preds"] for o in self.val_outputs])
        all_targets = np.concatenate([o["targets"] for o in self.val_outputs])
        log_visualizations(all_preds, all_targets, self.id2label)
        self.val_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        train_dataset = DogDataset(Path("data/processed/metadata.csv"), "train")
        return torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = DogDataset(Path("data/processed/metadata.csv"), "eval")
        return torch.utils.data.DataLoader(val_dataset, self.batch_size, shuffle=False)

    def test_dataloader(self):
        test_dataset = DogDataset(Path("data/processed/metadata.csv"), "test")
        return torch.utils.data.DataLoader(test_dataset, self.batch_size, shuffle=True)


if __name__ == "__main__":
    model = DogModel()
