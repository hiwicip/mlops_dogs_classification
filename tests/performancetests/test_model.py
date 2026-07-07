import os
import time

import torch
import wandb

from src.dogs_classification.model import DogModel

LOGDIR = "logs/performance"


def load_model(model_checkpoint: str) -> DogModel:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    api = wandb.Api(
        api_key=os.environ["WANDB_API_KEY"],
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=LOGDIR)
    file_name = artifact.files()[0].name
    model = DogModel()
    state_dict = torch.load(f"{LOGDIR}/{file_name}", map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model.model(torch.rand(1, 3, 224, 224))
    end = time.time()
    assert end - start < 100


# def test_model_accuracy():
#     model = load_model(os.getenv("MODEL_NAME"))
#     dataset = DogDataset(Path("data/processed/metadata.csv"), split="test")
#     loader = DataLoader(dataset, batch_size=32, shuffle=False)
#     acc = model.evaluate(loader)
#     assert acc > 0.75
