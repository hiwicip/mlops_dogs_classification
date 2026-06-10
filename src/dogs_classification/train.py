import hydra
import torch
from google.cloud import storage
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, schedule

from dogs_classification.model import DogModel

BUCKET_NAME = "mlops-dogs-data-euwest4"


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(cfg: DictConfig):
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs

    model = DogModel(model_name=cfg.model.name, lr=lr, batch_size=batch_size)

    wandb_logger = WandbLogger(
        project="dogs-classification",
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
            record_shapes=True,
            with_stack=True,
        )
        if cfg.training.profile
        else None
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="best_model",
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
    torch.save(checkpoint["state_dict"], "models/best_model.pt")

    client = storage.Client(project="mlopsdogclassification")
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("models/best_model.pt")
    blob.upload_from_filename("models/best_model.pt")

    print("Uploaded results to Bucket.")


if __name__ == "__main__":
    train()
