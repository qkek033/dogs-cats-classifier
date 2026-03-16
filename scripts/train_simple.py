"""
Train Dog vs Cat classifier. Run from project root: python scripts/train_simple.py
Uses config/config.yaml and config/training_config.yaml.
"""
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import timm
import mlflow
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

with open("config/config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
with open("config/training_config.yaml", encoding="utf-8") as f:
    training_config = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)


class DogCatDataset(torch.utils.data.Dataset):
    """Dataset over data/raw/train/cats and data/raw/train/dogs."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {"cats": 0, "dogs": 1}
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        logger.info("Loaded %d images from %s", len(self.images), root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = DogCatDataset("data/raw/train", transform=train_transform)
train_size = int(len(full_dataset) * 0.8)
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
for idx in range(len(val_dataset)):
    val_dataset.dataset.transform = val_transform
    break

batch_size = training_config["training"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

backbone = config["model"]["backbone"]
model = timm.create_model(backbone, pretrained=True, num_classes=2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=training_config["training"]["learning_rate"],
    weight_decay=training_config["training"]["weight_decay"],
)
scheduler = CosineAnnealingLR(optimizer, T_max=training_config["scheduler"]["t_max"])

mlflow.set_tracking_uri(training_config["mlflow"]["backend_store_uri"])
mlflow.set_experiment(training_config["mlflow"]["experiment_name"])

with mlflow.start_run():
    mlflow.log_params({
        "backbone": backbone,
        "epochs": training_config["training"]["epochs"],
        "batch_size": batch_size,
        "learning_rate": training_config["training"]["learning_rate"],
    })
    best_acc = 0.0
    patience_counter = 0
    epochs = training_config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc="Epoch %d/%d [TRAIN]" % (epoch + 1, epochs))
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({"loss": loss.item()})
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            "Epoch %d - Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f",
            epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc,
        )
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
        }, step=epoch)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, "models/checkpoints/best_model.pth")
            logger.info("Best model saved with accuracy: %.4f", best_acc)
            mlflow.pytorch.log_model(model, "best_model")
        else:
            patience_counter += 1

        if patience_counter >= training_config["early_stopping"]["patience"]:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    mlflow.log_metric("best_val_acc", best_acc)
    logger.info("Training completed. Best accuracy: %.4f", best_acc)
