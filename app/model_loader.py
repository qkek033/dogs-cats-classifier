"""Inference with optional OOD detection using backbone embeddings and class centroids."""
import torch
import timm
from pathlib import Path
import numpy as np
from PIL import Image
import time
import logging
import yaml

from app.ood_utils import (
    load_centroids,
    extract_embedding,
    detect_ood,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)

# Default paths (relative to project root when running from project root)
DEFAULT_CENTROIDS_PATH = "models/centroids.npz"


def load_model(checkpoint_path, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded from %s, val_acc=%s", checkpoint_path, checkpoint.get("val_acc", "?"))
    return model


def preprocess(image):
    """이미지 전처리. 모델이 float32이므로 입력도 float32로 통일."""
    if isinstance(image, Image.Image):
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
    else:
        image_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(torch.float32)


UNKNOWN_MESSAGE = "The uploaded image does not appear to contain a dog or cat."


def predict(model, image, device="cuda", centroids=None, ood_threshold=0.5, confidence_threshold=0.75):
    """
    예측 수행. 유효 출력은 dog, cat, unknown 뿐.
    - OOD(센트로이드 거리) 또는 신뢰도 < confidence_threshold 이면 unknown 반환.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    x = preprocess(image)
    x = x.to(device=device, dtype=torch.float32)

    t0 = time.time()

    # 1) OOD: embedding 기반 센트로이드 거리 검사
    if centroids is not None and ood_threshold is not None:
        embedding = extract_embedding(model, x, device)
        if detect_ood(embedding, centroids, ood_threshold, distance_method="cosine"):
            elapsed_ms = (time.time() - t0) * 1000
            return {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
                "inference_time_ms": elapsed_ms,
                "status": "unknown_detected",
                "message": UNKNOWN_MESSAGE,
            }

    # 2) 일반 분류
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    elapsed_ms = (time.time() - t0) * 1000
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    label = CLASS_NAMES[pred]

    # 3) 신뢰도 임계값 미만이면 unknown (landscape, random object 등 거부)
    if confidence < confidence_threshold:
        return {
            "label": "unknown",
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "inference_time_ms": elapsed_ms,
            "status": "unknown_detected",
            "message": UNKNOWN_MESSAGE,
        }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "inference_time_ms": elapsed_ms,
    }


class InferenceEngine:
    """훈련된 모델 추론 엔진 (OOD 지원)."""

    def __init__(self, checkpoint_path, device="cuda", config=None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = load_model(checkpoint_path, self.device)

        self.centroids = None
        self.ood_threshold = None
        self.confidence_threshold = 0.75

        if config:
            inf = config.get("inference", {})
            self.ood_threshold = inf.get("ood_distance_threshold")
            self.confidence_threshold = float(inf.get("confidence_threshold", 0.75))
            centroids_path = inf.get("centroids_path", DEFAULT_CENTROIDS_PATH)
            if isinstance(checkpoint_path, str):
                # checkpoint: models/checkpoints/best_model.pth -> project root = parent.parent.parent
                base = Path(checkpoint_path).resolve().parent.parent.parent
            else:
                base = Path(".").resolve()
            path = base / centroids_path if not Path(centroids_path).is_absolute() else Path(centroids_path)
            self.centroids = load_centroids(path)
            if self.centroids is None and self.ood_threshold is not None:
                logger.warning("OOD threshold set but centroids not loaded; OOD detection disabled.")
        else:
            path = Path(checkpoint_path).resolve().parent.parent if isinstance(checkpoint_path, str) else Path(".")
            self.centroids = load_centroids(path / DEFAULT_CENTROIDS_PATH)
            self.ood_threshold = 0.5

    def predict(self, image):
        return predict(
            self.model,
            image,
            device=self.device,
            centroids=self.centroids,
            ood_threshold=self.ood_threshold,
            confidence_threshold=self.confidence_threshold,
        )
