import os
from pathlib import Path

import httpx
import pytest

TESTS_DIR = Path(__file__).parent
TEST_IMAGE = TESTS_DIR / "data" / "n02097130_4518.jpg"

BASE_URL = os.environ.get("BENTOML_URL", "https://dogs-bentoml-288634047169.europe-west4.run.app")


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=120) as c:
        # warm up: Cloud Run may be cold-starting, retry for up to 3 minutes
        for attempt in range(9):
            try:
                r = c.get("/livez")
                if r.status_code == 200:
                    break
            except httpx.TransportError:
                if attempt == 8:
                    raise
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
    predictions = response.json()["predictions"]
    assert all(f"class{i}" in predictions for i in range(1, 6))
    assert all(f"confidence{i}" in predictions for i in range(1, 6))


def test_predict_confidence_in_range(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"image": ("dog.jpg", f, "image/jpeg")})
    predictions = response.json()["predictions"]
    assert all(0.0 <= predictions[f"confidence{i}"] <= 1.0 for i in range(1, 6))


def test_predict_class_is_string(client):
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict", files={"image": ("dog.jpg", f, "image/jpeg")})
    predictions = response.json()["predictions"]
    assert all(isinstance(predictions[f"class{i}"], str) for i in range(1, 6))
