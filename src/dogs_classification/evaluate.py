import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from dogs_classification.model import DogModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUCKET_NAME = "mlops-dog-data-euwest4"


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def evaluate(cfg: DictConfig):
    model = DogModel(model_name=cfg.model.name, batch_size=cfg.training.batch_size)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))

    wandb_logger = WandbLogger(
        project="dogs-classification",
        entity="awinterstetter",
        job_type="evaluation",
        config={
            "learning_rate": cfg.training.lr,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "model_name": cfg.model.name,
        },
    )

    trainer = Trainer(
        logger=wandb_logger,
    )
    trainer.test(model)

    # Save evaluation results as JSON
    # output_path = Path("logs/eval/performance.json")
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    # with open(output_path, "w") as f:
    #    json.dump(result, f, indent=2)

    # Upload results JSON file to GCS bucket
    # client = storage.Client(project="mlopsdogclassification")
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob("logs/eval/performance.json")
    # blob.upload_from_filename(output_path)

    # print("Uploaded results to Bucket.")


if __name__ == "__main__":
    evaluate()
