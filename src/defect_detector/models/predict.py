"""Loads the trained model once and exposes a simple predict function for a
single image. Kept separate from api/main.py so this logic is testable
without spinning up FastAPI, same pattern as Project 1.
"""
import torch
from functools import lru_cache
from pathlib import Path
from PIL import Image

from defect_detector.data.load import build_transforms
from defect_detector.models.model import build_model
from defect_detector.utils.config import load_config, resolve


@lru_cache(maxsize=1)
def get_model():
    cfg = load_config()
    model_path = resolve(cfg["model"]["save_path"])
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train it first (see Colab notebook)."
        )

    model = build_model(cfg["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, cfg


def predict_image(image: Image.Image) -> dict:
    model, cfg = get_model()

    transform = build_transforms(cfg["data"]["image_size"], train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = int(probabilities.argmax())

    classes = cfg["data"]["class_names"]
    return {
        "predicted_class": classes[predicted_idx],
        "confidence": round(float(probabilities[predicted_idx]), 4),
        "probabilities": {classes[i]: round(float(p), 4) for i, p in enumerate(probabilities)},
    }