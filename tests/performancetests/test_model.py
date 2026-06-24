import os
import time

import torch
import wandb

from src.dogs_classification.model import DogModel

LOGDIR = "logs/performance"


def load_model(model_checkpoint: str) -> DogModel:
    api = wandb.Api(
        api_key=os.environ["WANDB_API_KEY"],
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=LOGDIR)
    file_name = artifact.files()[0].name
    return DogModel.load_from_checkpoint(f"{LOGDIR}/{file_name}")


def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model.model(torch.rand(1, 3, 224, 224))
    end = time.time()
    assert end - start < 100
