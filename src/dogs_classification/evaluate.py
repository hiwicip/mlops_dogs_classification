from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dogs_classification.data import DogDataset
from dogs_classification.model import DogModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BUCKET_NAME = "mlops-dogs-data-euwest"


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def evaluate(cfg: DictConfig):
    model = DogModel(model_name=cfg.model.name)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    eval_dataset = DogDataset(Path("data/processed/metadata.csv"), "eval")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            img = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(img).logits
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    correct_per_class: dict[int, int] = defaultdict(int)
    total_per_class: dict[int, int] = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        total_per_class[label] += 1
        if pred == label:
            correct_per_class[label] += 1

    class_accuracy = {int(cls): correct_per_class[cls] / total_per_class[cls] for cls in total_per_class}

    result = {"accuracy": float(acc), "class_accuracy": class_accuracy}

    print("Evaluated the thing")
    print(result)

    output_dir = Path("logs/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "all_preds.npy", np.array(all_preds))
    np.save(output_dir / "all_labels.npy", np.array(all_labels))

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
