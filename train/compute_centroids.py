"""
Compute class centroids from training data using backbone features.
Run after training. Saves models/centroids.npz for OOD detection.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["cat", "dog"]
# 폴더명 (데이터 디렉토리: cats, dogs)
FOLDER_NAMES = ["cats", "dogs"]


def preprocess(image):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if isinstance(image, Image.Image):
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0
    else:
        arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)


class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.labels = []
        for i, folder in enumerate(FOLDER_NAMES):
            d = self.root_dir / folder
            if not d.exists():
                continue
            for p in d.glob("*.jpg"):
                self.samples.append(str(p))
                self.labels.append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        x = preprocess(img).squeeze(0)
        return x, self.labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "models/checkpoints/best_model.pth"))
    parser.add_argument("--data-dir", type=str, default=str(ROOT / "data/raw/train"))
    parser.add_argument("--output", type=str, default=str(ROOT / "models/centroids.npz"))
    parser.add_argument("--max-per-class", type=int, default=2000, help="Max samples per class")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return

    logger.info("Loading model...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    logger.info("Loading dataset...")
    dataset = SimpleDataset(args.data_dir)
    if len(dataset) == 0:
        logger.error("No images found in %s", args.data_dir)
        return

    # Subsample per class
    by_class = {0: [], 1: []}
    for i in range(len(dataset)):
        idx = dataset.labels[i]
        by_class[idx].append(i)
    indices = []
    for idx in [0, 1]:
        arr = by_class[idx]
        if len(arr) > args.max_per_class:
            np.random.seed(42)
            arr = np.random.choice(arr, args.max_per_class, replace=False).tolist()
        indices.extend(arr)
    np.random.seed(42)
    np.random.shuffle(indices)

    # Accumulate embeddings per class
    embeddings_cat = []
    embeddings_dog = []
    for i in tqdm(indices, desc="Extracting embeddings"):
        x, label = dataset[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.forward_features(x)
            if feats.dim() == 4:
                emb = feats.mean(dim=[2, 3]).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                emb = feats.squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        if label == 0:
            embeddings_cat.append(emb)
        else:
            embeddings_dog.append(emb)

    centroid_cat = np.stack(embeddings_cat, axis=0).mean(axis=0).astype(np.float32)
    centroid_dog = np.stack(embeddings_dog, axis=0).mean(axis=0).astype(np.float32)
    for vec in (centroid_cat, centroid_dog):
        n = np.linalg.norm(vec)
        if n > 1e-8:
            vec /= n

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, cat=centroid_cat, dog=centroid_dog)
    logger.info("Saved centroids to %s", out_path)


if __name__ == "__main__":
    main()
