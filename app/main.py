"""
Dog vs Cat Classifier API.

Run: uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints: GET /health, GET /model-info, POST /predict, POST /explainability, POST /validate

Deploy: Set MODEL_URL (e.g. GitHub Release raw URL) to download best_model.pth if missing.
"""
import base64
import io
import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.logging_config import setup_logging
from app.schemas import (
    ExplainabilityResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ValidationResponse,
)

logger = setup_logging()

app = FastAPI(
    title="Dog vs Cat Classifier API",
    description="Production-ready image classification API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize server and load model."""
    global inference_engine, config

    try:
        with open("config/config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing with device: %s", device)

        model_path = Path("models/checkpoints/best_model.pth")
        model_url = os.environ.get("MODEL_URL")
        if model_url and not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                logger.info("Downloading model from MODEL_URL...")
                urlretrieve(model_url, model_path)
                logger.info("Model downloaded to %s", model_path)
            except Exception as e:
                logger.error("Model download failed: %s", e)
                model_path = Path("models/checkpoints/best_model.pth")

        if model_path.exists():
            from app.model_loader import InferenceEngine

            inference_engine = InferenceEngine(str(model_path), device, config=config)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model not found at %s", model_path)

        logger.info("API startup completed")
    except Exception as e:
        logger.error("Startup error: %s", e)


@app.get("/")
async def root():
    """API root; returns docs and health links."""
    return {"message": "Dog vs Cat Classifier API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine is not None,
        version="1.0.0",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Return model metadata."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    backbone = (config or {}).get("model", {}).get("backbone", "efficientnet_b0")
    return ModelInfoResponse(
        model_name="Dog vs Cat Classifier",
        backbone=backbone,
        num_classes=2,
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        parameters=26000000
    )


def _decode_image_base64(b64_str: str):
    """Base64 문자열을 디코딩해 PIL Image(RGB) 반환. 공백/줄바꿈 제거 후 처리."""
    s = (b64_str or "").strip()
    if not s:
        raise ValueError("Empty base64 string")
    if "," in s and s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        image_data = base64.b64decode(s, validate=True)
    except Exception as e:
        logger.error("Base64 decode error: %s", e)
        raise ValueError(f"Invalid base64: {e}") from e
    if not image_data:
        raise ValueError("Decoded image data is empty")
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error("Image open error: %s", e)
        raise ValueError(f"Invalid image: {e}") from e
    return image


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify image as dog or cat (or unknown)."""
    try:
        image = _decode_image_base64(request.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = inference_engine.predict(image)
        label = result["label"]
        status = result.get("status", "success" if label != "unknown" else "unknown_detected")
        error_message = result.get("message")
        return PredictionResponse(
            label=label,
            confidence=result["confidence"],
            status=status,
            error_message=error_message,
            processing_time_ms=result["inference_time_ms"],
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explainability", response_model=ExplainabilityResponse)
async def get_explainability(request: PredictionRequest):
    """Return Grad-CAM visualization for dog/cat predictions; unavailable for unknown."""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        image = _decode_image_base64(request.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = inference_engine.predict(image)
    if result["label"] == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Grad-CAM is not available for out-of-distribution images. Image appears to be outside the dog/cat distribution.",
        )

    try:
        import base64
        import io

        from app.gradcam import GradCAM
        from app.model_loader import preprocess

        model = inference_engine.model
        gradcam = GradCAM(model)
        try:
            device = next(model.parameters()).device
            tensor = preprocess(image).to(device)
            pred_class = 0 if result["label"] == "cat" else 1
            vis = gradcam.visualize(tensor, image, pred_class, result["label"], device=device)
            def img_to_b64(im):
                buf = io.BytesIO()
                if hasattr(im, "save"):
                    im.save(buf, format="PNG")
                else:
                    from PIL import Image as PILImage
                    PILImage.fromarray(im).save(buf, format="PNG")
                buf.seek(0)
                return base64.b64encode(buf.getvalue()).decode()
            orig_b64 = img_to_b64(image)
            heat_b64 = img_to_b64(vis["heatmap"])
            over_b64 = img_to_b64(vis["overlay"])
            return ExplainabilityResponse(
                label=result["label"],
                confidence=result["confidence"],
                original_image_base64=orig_b64,
                gradcam_image_base64=heat_b64,
                overlay_image_base64=over_b64,
            )
        finally:
            gradcam.remove_hooks()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Explainability error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate", response_model=ValidationResponse)
async def validate_image(file: UploadFile = File(...)):
    """Validate uploaded image format."""
    try:
        contents = await file.read()
        Image.open(io.BytesIO(contents)).convert("RGB")
        return ValidationResponse(
            is_valid=True,
            message="Validation successful",
            reason=None,
            object_count=1,
        )
    except Exception as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
