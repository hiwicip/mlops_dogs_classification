import pytest
import torch

from src.dogs_classification.model import DogModel


def test_model_instantiation():
    """Model should build without errors."""
    model = DogModel()
    assert model is not None, "DogModel() returned None"


def test_model_output_shape():
    """Forward pass output shape check."""
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    out = model.model(x).logits
    assert out.shape == (4, 80), f"Expected output shape (4, 80), got {tuple(out.shape)}"


def test_model_output_dtype():
    """Output should be float32."""
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    out = model.model(x).logits
    assert out.dtype == torch.float32, f"Expected float32 output, got {out.dtype}"


def test_model_no_nan_output():
    """Forward pass should not produce NaN."""
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    out = model.model(x).logits
    assert not torch.isnan(out).any(), "Model output contains NaN values"


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_model_different_batch_sizes(batch_size):
    """Model should handle any batch size."""
    model = DogModel()
    out = model.model(torch.randn(batch_size, 3, 224, 224)).logits
    assert out.shape[0] == batch_size, f"Expected batch size {batch_size} in output, got {out.shape[0]}"


def test_model_parameter_count():
    """Check model size is as expected."""
    model = DogModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 85_860_176, f"Expected 85,860,176 parameters, got {n_params:,}"


def test_model_gradients():
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 80, (4,))
    loss = torch.nn.functional.cross_entropy(model.model(x).logits, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


@pytest.mark.parametrize("lr", [0, -0.1, -1e-5])
def test_invalid_lr_raises(lr):
    """Non-positive learning rate should raise ValueError."""
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        DogModel(lr=lr)


@pytest.mark.parametrize("batch_size", [0, -1, -32])
def test_invalid_batch_size_raises(batch_size):
    """Batch size below 1 should raise ValueError."""
    with pytest.raises(ValueError, match="Batch size must be at least 1"):
        DogModel(batch_size=batch_size)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 1, 28, 28),  # wrong channels and spatial size
        (4, 3, 112, 112),  # correct channels, wrong spatial size
        (4, 3, 224),  # missing spatial dimension (3D instead of 4D)
    ],
)
def test_invalid_input_shape_raises(shape):
    """Wrong input shape to training_step should raise ValueError."""
    model = DogModel()
    model.eval()
    batch = {"pixel_values": torch.randn(*shape), "labels": torch.randint(0, 80, (4,)), "breed": ["x"] * 4}
    with pytest.raises(ValueError, match="Expected input shape"):
        model.training_step(batch, batch_idx=0)
