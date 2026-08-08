"""FastAPI app serving the defect classifier.

Run with:  uv run uvicorn defect_detector.api.main:app --reload
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from defect_detector.models.predict import predict_image
from defect_detector.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Casting Defect Detector API",
    description="Classifies casting product images as defective or ok.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_image(image)
        return result
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")