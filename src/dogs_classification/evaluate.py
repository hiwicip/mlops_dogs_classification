from pathlib import Path

import torch
import typer
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from dogs_classification.model import DogModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUCKET_NAME = "mlops-dog-data-euwest4"

WANDB_ENTITY = "awinterstetter"
WANDB_PROJECT = "dogs-classification"

app = typer.Typer()


@app.command()
def evaluate(artifact_name: str = "best_model:latest"):
    """
    Evaluate the dog classification model on the test set.
    Args:
        artifact_name (str): The name of the W&B artifact to evaluate.
    Returns:
        None
    """

    api = wandb.Api()

    artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}")

    filename = next(iter(artifact.manifest.entries.values())).path

    checkpoint_path = Path("models") / filename

    if checkpoint_path.exists():
        print("Using local model:", checkpoint_path)
    else:
        print("Downloading model from W&B artifact:", artifact_name)
        artifact.download(root="models")
        download_dir = Path(artifact.download(root="models"))
        checkpoint_path = download_dir / filename

    run = artifact.logged_by()
    config = run.config  # get the config from the run that logged the artifact

    model_name = config["model_name"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    epochs = config["epochs"]

    model = DogModel(model_name=model_name, batch_size=batch_size)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="evaluation",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "model_name": model_name,
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
    app()
