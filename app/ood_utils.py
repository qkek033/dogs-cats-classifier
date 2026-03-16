"""
Out-of-Distribution detection using feature embeddings and class centroids.
"""
import numpy as np
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ["cat", "dog"]


def extract_embedding(model, image_tensor, device="cuda"):
    """
    Extract feature vector from model backbone (before classification layer).
    Uses forward_features + global average pooling.
    Args:
        model: timm model (e.g. efficientnet_b0)
        image_tensor: (1, 3, H, W) tensor on same device as model
    Returns:
        np.ndarray: shape (D,) normalized embedding
    """
    model.eval()
    with torch.no_grad():
        features = model.forward_features(image_tensor)
        if features.dim() == 4:
            embedding = features.mean(dim=[2, 3]).squeeze(0)
        else:
            embedding = features.squeeze(0)
        embedding = embedding.cpu().numpy().astype(np.float32)
    # L2 normalize for cosine distance
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding


def load_centroids(path):
    """
    Load class centroids from file.
    Expected format: np.savez(path, cat=array, dog=array) with L2-normalized vectors.
    Returns:
        dict[str, np.ndarray] or None if file missing/invalid
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Centroids file not found: %s", path)
        return None
    try:
        data = np.load(path)
        centroids = {}
        for name in CLASS_NAMES:
            if name not in data:
                logger.warning("Centroid '%s' not in %s", name, path)
                return None
            vec = data[name].astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            centroids[name] = vec
        logger.info("Loaded centroids from %s", path)
        return centroids
    except Exception as e:
        logger.warning("Failed to load centroids from %s: %s", path, e)
        return None


def compute_distance(embedding, centroid, method="cosine"):
    """
    Distance between embedding and centroid.
    Both should be L2-normalized for cosine.
    Args:
        embedding: (D,) array
        centroid: (D,) array
    Returns:
        float: distance (cosine distance = 1 - cos_sim, or euclidean)
    """
    if method == "cosine":
        # cosine distance = 1 - cosine_similarity (assuming normalized)
        sim = np.dot(embedding, centroid)
        sim = np.clip(sim, -1.0, 1.0)
        return float(1.0 - sim)
    else:
        return float(np.linalg.norm(embedding - centroid))


def detect_ood(embedding, centroids, threshold, distance_method="cosine"):
    """
    Determine if embedding is out-of-distribution.
    If min distance to any class centroid is larger than threshold, treat as OOD.
    Args:
        embedding: (D,) normalized array
        centroids: dict with keys 'cat', 'dog', values (D,) normalized arrays
        threshold: float, ood_distance_threshold from config
        distance_method: "cosine" or "euclidean"
    Returns:
        bool: True if OOD (should return unknown)
    """
    if not centroids:
        return False
    min_dist = float("inf")
    for name in CLASS_NAMES:
        if name not in centroids:
            return False
        d = compute_distance(embedding, centroids[name], method=distance_method)
        min_dist = min(min_dist, d)
    return bool(min_dist > threshold)
