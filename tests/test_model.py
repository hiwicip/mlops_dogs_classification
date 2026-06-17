import torch

from src.dogs_classification.model import DogModel


def test_model_instantiation():
    """Model should build without errors."""
    model = DogModel()
    assert model is not None


def test_model_output_shape():
    """Forward pass output shape check."""
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    out = model.model(x).logits
    assert out.shape == (4, 80)


def test_model_output_dtype():
    """Output should be float32."""
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    out = model.model(x).logits
    assert out.dtype == torch.float32


def test_model_no_nan_output():
    """Forward pass should not produce NaN."""
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    out = model.model(x).logits
    assert not torch.isnan(out).any()


def test_model_different_batch_sizes():
    """Model should handle any batch size."""
    model = DogModel()
    for bs in [1, 8, 32, 64]:
        x = torch.randn(bs, 3, 224, 224)
        out = model.model(x).logits
        assert out.shape[0] == bs


def test_model_parameter_count():
    """Check model size is as expected."""
    model = DogModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 85_860_176


def test_model_gradients():
    model = DogModel()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 80, (4,))
    loss = torch.nn.functional.cross_entropy(model.model(x).logits, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
