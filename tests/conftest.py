from pathlib import Path
from unittest.mock import patch

import pytest

TESTS_DIR = Path(__file__).parent


@pytest.fixture(autouse=True)
def patch_classes_file():
    """Point DogModel at the committed test classes.json instead of the DVC-tracked one."""
    with patch("dogs_classification.model.CLASSES_FILE", TESTS_DIR / "classes.json"):
        yield
