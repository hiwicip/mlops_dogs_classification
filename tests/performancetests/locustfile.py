import io

from locust import HttpUser, between, task
from PIL import Image


def make_image_bytes():
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class DogFanatic(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_livez(self) -> None:
        """A task that hits the BentoML service's liveness endpoint."""
        self.client.get("/livez")

    @task(3)
    def predict(self) -> None:
        self.client.post(
            "/predict",
            files={"image": ("dog.jpg", make_image_bytes(), "image/jpeg")},
            headers={"X-Test-Request": "true"},
        )
