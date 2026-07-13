from datetime import datetime

import hydra
import torch
import wandb
from google.cloud import storage
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, schedule

from dogs_classification.model import DogModel

BUCKET_NAME = "mlops-dog-data-euwest4"


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(cfg: DictConfig):
    """
    Train the dog classification model.
    Args:
        cfg (DictConfig): The configuration object containing training parameters.
    Returns:
        None
    """
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs

    model = DogModel(model_name=cfg.model.name, lr=lr, batch_size=batch_size)

    wandb_logger = WandbLogger(
        project="dogs-classification",
        entity="awinterstetter",
        job_type="training",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "model_name": cfg.model.name,
        },
    )

    profiler = (
        PyTorchProfiler(
            dirpath="./logs/profiler",
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            profile_memory=True,
            # record_shapes=True,
            # with_stack=True,
        )
        if cfg.training.profile
        else None
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename=f"best_model_{timestamp}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        profiler=profiler,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

    checkpoint = torch.load(checkpoint_callback.best_model_path)
    model_path = f"models/best_model_{timestamp}.pt"
    torch.save(checkpoint["state_dict"], model_path)

    client = storage.Client(project="mlopsdogclassification")
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/best_model_{timestamp}.pt")
    blob.upload_from_filename(model_path)
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_reference(f"gs://{BUCKET_NAME}/models/best_model_{timestamp}.pt")
    logged_artifact = wandb_logger.experiment.log_artifact(artifact)
    wandb_logger.experiment.link_artifact(
        artifact=logged_artifact,
        target_path="wandb-registry-dog_models/best_models",
        aliases=["latest", "staging"],
    )

    print("Uploaded model to GCS bucket and logged to WandB model registry.")


if __name__ == "__main__":
    train()
