from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from src.dogs_classification.model import DogModel


class _SyntheticDogDataset(Dataset):
    """In-memory dataset matching the dict batch format DogModel expects."""

    def __init__(self, n_samples: int = 8, n_classes: int = 80):
        self.images = torch.randn(n_samples, 3, 224, 224)
        self.labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        return {
            "pixel_values": self.images[idx],
            "labels": self.labels[idx],
            "breed": f"breed_{self.labels[idx].item()}",
        }


@pytest.fixture
def model():
    """DogModel with synthetic in-memory dataloaders — no disk or network needed."""
    m = DogModel()
    loader = DataLoader(_SyntheticDogDataset(), batch_size=4)
    m.train_dataloader = lambda: loader
    m.val_dataloader = lambda: loader
    return m


# --- Training behaviour (exercises the Trainer pipeline, like train.py does) ---


def test_training_step_runs(model):
    """A single training + validation step via Trainer should complete without error."""
    trainer = Trainer(max_epochs=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(model)


# def test_loss_decreases(model):
#    """Training loss should decrease when repeatedly fitting the same single batch."""
#    trainer = Trainer(max_epochs=5, overfit_batches=1, logger=False, enable_checkpointing=False)
#    trainer.fit(model)
#    # Random baseline for 80 classes ≈ log(80) ≈ 4.38; after 5 steps it must be lower
#    assert (
#        trainer.callback_metrics["train_loss"].item() < torch.log(torch.tensor(80.0)).item()
#    ), "Loss did not decrease below random baseline after 5 epochs on a fixed batch"


def test_parameters_updated(model):
    """At least one parameter tensor should change after a single Trainer step."""
    params_before = [p.data.clone() for p in model.parameters()]
    trainer = Trainer(max_epochs=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(model)
    params_after = [p.data for p in model.parameters()]
    assert any(
        not torch.equal(before, after) for before, after in zip(params_before, params_after, strict=True)
    ), "No parameters changed after a training step — optimizer may not be updating weights"


# def test_overfit_one_batch(model):
#    """Model should memorise a single batch — confirms sufficient model capacity."""
#    trainer = Trainer(max_epochs=30, overfit_batches=1, logger=False, enable_checkpointing=False)
#    trainer.fit(model)
#    final_loss = trainer.callback_metrics["train_loss"].item()
#    random_loss = torch.log(torch.tensor(80.0)).item()
#    assert (
#        final_loss < random_loss * 0.5
#    ), f"Model failed to overfit: final={final_loss:.4f}, random baseline={random_loss:.4f}"


# --- train.py structure tests (verify setup without running real training) ---


@pytest.fixture
def hydra_cfg():
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../configs"):
        yield compose(config_name="config.yaml", overrides=["training.profile=False"])
    GlobalHydra.instance().clear()


def _make_mock_trainer(captured_callbacks: list):
    def factory(*args, **kwargs):
        captured_callbacks.extend(kwargs.get("callbacks", []))
        t = MagicMock()
        t.checkpoint_callbacks = [MagicMock(best_model_path="models/best.ckpt")]
        return t

    return factory


def test_checkpoint_monitors_val_loss(hydra_cfg):
    """ModelCheckpoint in train.py must monitor val_loss in min mode with save_top_k=1."""
    from src.dogs_classification.train import train

    captured: list = []
    with (
        patch("src.dogs_classification.train.WandbLogger") as mock_wl,
        patch("src.dogs_classification.train.Trainer", side_effect=_make_mock_trainer(captured)),
        patch("src.dogs_classification.train.storage.Client"),
        patch("src.dogs_classification.train.torch.save"),
        patch("src.dogs_classification.train.torch.load", return_value={"state_dict": {}}),
        patch("src.dogs_classification.train.DogModel"),
    ):
        mock_wl.return_value.experiment = MagicMock()
        train.__wrapped__(hydra_cfg)

    checkpoint_cb = next(c for c in captured if isinstance(c, ModelCheckpoint))
    assert checkpoint_cb.monitor == "val_loss", f"Checkpoint should monitor 'val_loss', got '{checkpoint_cb.monitor}'"
    assert checkpoint_cb.mode == "min", f"Checkpoint mode should be 'min', got '{checkpoint_cb.mode}'"
    assert (
        checkpoint_cb.save_top_k == 1
    ), f"Checkpoint should save top 1 model, got save_top_k={checkpoint_cb.save_top_k}"


def test_gcs_upload_called_after_training(hydra_cfg):
    """Model file should be uploaded to GCS after trainer.fit completes."""
    from src.dogs_classification.train import train

    mock_blob = MagicMock()
    mock_client = MagicMock()
    mock_client.bucket.return_value.blob.return_value = mock_blob

    with (
        patch("src.dogs_classification.train.WandbLogger") as mock_wl,
        patch("src.dogs_classification.train.Trainer", side_effect=_make_mock_trainer([])),
        patch("src.dogs_classification.train.storage.Client", return_value=mock_client),
        patch("src.dogs_classification.train.torch.save"),
        patch("src.dogs_classification.train.torch.load", return_value={"state_dict": {}}),
        patch("src.dogs_classification.train.DogModel"),
    ):
        mock_wl.return_value.experiment = MagicMock()
        train.__wrapped__(hydra_cfg)

    mock_blob.upload_from_filename.assert_called_once()
