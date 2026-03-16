"""
Inference pipeline with optional Out-of-Distribution (OOD) detection.
Uses feature embeddings and class centroids for robust unknown rejection.
"""
import torch
import numpy as np
from PIL import Image
import time
import logging
import yaml
from pathlib import Path

from app.ood_utils import (
    load_centroids,
    extract_embedding,
    compute_distance,
    detect_ood,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CENTROIDS_PATH = "models/centroids.npz"

UNKNOWN_MESSAGE = "The uploaded image does not appear to contain a dog or cat."


def preprocess(image):
    """Normalize image to tensor (1, 3, 224, 224) float32."""
    if isinstance(image, Image.Image):
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0
    else:
        arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(torch.float32)


def load_model(checkpoint_path, device="cuda"):
    """Load timm efficientnet_b0 from checkpoint."""
    import timm
    device = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def inference_with_ood(
    model,
    image,
    device="cuda",
    centroids=None,
    ood_distance_threshold=0.5,
    confidence_threshold=0.75,
):
    """
    Single-image inference. Valid outputs: dog, cat, unknown.
    1. OOD: if centroids set and min distance > threshold -> unknown.
    2. Classification: softmax -> pred_class, confidence.
    3. If confidence < confidence_threshold -> unknown.
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    x = preprocess(image).to(device=device, dtype=torch.float32)
    t0 = time.time()

    if centroids is not None and ood_distance_threshold is not None:
        emb = extract_embedding(model, x, device)
        if detect_ood(emb, centroids, ood_distance_threshold, distance_method="cosine"):
            elapsed = (time.time() - t0) * 1000
            return {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
                "inference_time_ms": elapsed,
                "status": "unknown_detected",
                "message": UNKNOWN_MESSAGE,
            }

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed = (time.time() - t0) * 1000
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    label = CLASS_NAMES[pred]

    if confidence < confidence_threshold:
        return {
            "label": "unknown",
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "inference_time_ms": elapsed,
            "status": "unknown_detected",
            "message": UNKNOWN_MESSAGE,
        }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "inference_time_ms": elapsed,
    }


class PyTorchOODInferenceEngine:
    """
    PyTorch inference with embedding-based OOD detection.
    Uses config for ood_distance_threshold and centroids_path.
    """

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
            cp = inf.get("centroids_path", DEFAULT_CENTROIDS_PATH)
            base = Path(checkpoint_path).resolve().parent.parent if isinstance(checkpoint_path, str) else Path(".")
            path = base / cp if not Path(cp).is_absolute() else Path(cp)
            self.centroids = load_centroids(path)
        else:
            base = Path(checkpoint_path).resolve().parent.parent if isinstance(checkpoint_path, str) else Path(".")
            self.centroids = load_centroids(base / DEFAULT_CENTROIDS_PATH)
            self.ood_threshold = 0.5

    def predict(self, image):
        return inference_with_ood(
            self.model,
            image,
            device=self.device,
            centroids=self.centroids,
            ood_distance_threshold=self.ood_threshold,
            confidence_threshold=self.confidence_threshold,
        )


def create_inference_engine(model_type, model_path, config_path=None, device="cuda"):
    """Create inference engine. For pytorch with OOD use PyTorchOODInferenceEngine."""
    if model_type == "pytorch":
        config = None
        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        return PyTorchOODInferenceEngine(model_path, device, config=config)
    raise ValueError(f"Unsupported model type: {model_type}")
