import io
from PIL import Image
from fastapi.testclient import TestClient
import defect_detector.api.main as api_module

client = TestClient(api_module.app)


def _fake_image_bytes():
    """Build a tiny real image in memory, so we have something valid to upload."""
    img = Image.new("RGB", (128, 128), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_shape(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "predict_image",
        lambda image: {
            "predicted_class": "ok_front",
            "confidence": 0.95,
            "probabilities": {"def_front": 0.05, "ok_front": 0.95},
        },
    )
    response = client.post(
        "/predict", files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_non_image_file():
    fake_text_file = io.BytesIO(b"this is not an image")
    response = client.post(
        "/predict", files={"file": ("test.txt", fake_text_file, "text/plain")}
    )
    assert response.status_code == 400