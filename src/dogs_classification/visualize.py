import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import wandb


def log_visualizations(preds: np.ndarray, labels: np.ndarray, idx_to_class: dict) -> None:
    num_classes = len(idx_to_class)

    cm = confusion_matrix(labels, preds, normalize="true")

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.colorbar()
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(
        ticks=np.arange(num_classes),
        labels=[idx_to_class[i] for i in range(num_classes)],
        rotation=90,
        fontsize=6,
    )
    plt.yticks(
        ticks=np.arange(num_classes),
        labels=[idx_to_class[i] for i in range(num_classes)],
        fontsize=6,
    )
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    for p, t in zip(preds, labels):
        total[t] += 1
        if p == t:
            correct[t] += 1

    acc_per_class = correct / np.maximum(total, 1)
    class_labels = [idx_to_class[i] for i in range(num_classes)]

    plt.figure(figsize=(14, 6))
    plt.bar(class_labels, acc_per_class)
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Dog Breed")
    plt.tight_layout()
    wandb.log({"accuracy_per_class": wandb.Image(plt)})
    plt.close()


def visualize():
    preds = np.load("logs/eval/all_preds.npy")
    labels = np.load("logs/eval/all_labels.npy")

    with open("data/processed/classes.json") as f:
        class_names = json.load(f)

    idx_to_class = {idx: name for name, idx in class_names.items()}

    wandb.init(project="dogs-classification", job_type="visualize")
    log_visualizations(preds, labels, idx_to_class)
    wandb.finish()


if __name__ == "__main__":
    visualize()
