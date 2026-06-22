from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TESTS_DIR = Path(__file__).parent


@pytest.fixture(autouse=True)
def patch_classes_file():
    """Point DogModel at the committed test classes.json instead of the DVC-tracked one."""
    with patch("dogs_classification.model.CLASSES_FILE", TESTS_DIR / "classes.json"):
        yield


@pytest.fixture(autouse=True)
def mock_wandb():
    """Prevent wandb.log/wandb.Image calls from failing when wandb is not initialised."""
    with patch("wandb.log"), patch("wandb.Image", return_value=MagicMock()):
        yield
