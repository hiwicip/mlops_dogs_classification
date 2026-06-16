import io

from fastapi.testclient import TestClient
from PIL import Image

from dogs_classification.api import app

TEST_IMAGE = "tests/data/n02097130_4518.jpg"

def make_image_bytes():
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Dogs Classification API!"}

def test_predict_no_file():
    with TestClient(app) as client:
        response = client.post("/predict")
        assert response.status_code == 422

def test_predict_invalid_file():
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
        assert response.status_code == 400

def test_predict_valid_image():
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("dog.jpg", make_image_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert "breed" in body
    assert "confidence" in body
    assert isinstance(body["breed"], str)
    assert 0.0 <= body["confidence"] <= 1.0

def test_predict_response_schema():
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("dog.jpg", make_image_bytes(), "image/jpeg")})
    assert set(response.json().keys()) == {"breed", "confidence"}

def test_predict_real_dog_image():
    with open(TEST_IMAGE, "rb") as f:
        with TestClient(app) as client:
            response = client.post("/predict", files={"file": ("test_dog.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["breed"], str)
    assert 0.0 <= body["confidence"] <= 1.0
