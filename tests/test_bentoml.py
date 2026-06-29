import os
from pathlib import Path

import httpx
import pytest

TESTS_DIR = Path(__file__).parent
TEST_IMAGE = TESTS_DIR / "data" / "n02097130_4518.jpg"

BASE_URL = os.environ.get(
    "BENTOML_URL", "https://dogs-bentoml-288634047169.europe-west4.run.app"
)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


def test_service_is_live(client):
    response = client.get("/livez")
    assert response.status_code == 200


def test_predict_response_status(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"image": ("dog.jpg", f, "image/jpeg")})
    assert response.status_code == 200


def test_predict_response_schema(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"image": ("dog.jpg", f, "image/jpeg")})
    body = response.json()
    assert set(body.keys()) == {"predicted_class", "confidence"}


def test_predict_confidence_in_range(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"image": ("dog.jpg", f, "image/jpeg")})
    assert 0.0 <= response.json()["confidence"] <= 1.0


def test_predict_class_is_string(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"image": ("dog.jpg", f, "image/jpeg")})
    assert isinstance(response.json()["predicted_class"], str)
