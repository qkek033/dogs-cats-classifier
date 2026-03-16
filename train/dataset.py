import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class DogCatDataset(Dataset):
    """Dogs vs Cats classification dataset"""
    
    def __init__(self, root_dir, split='train', transform=None, config_path='config/config.yaml'):
        """
        Args:
            root_dir: 데이터 루트 디렉토리
            split: 'train', 'val', 'test'
            transform: 이미지 변환
            config_path: 설정 파일 경로
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {'cats': 0, 'dogs': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 이미지 로드
        self._load_images()
        
        if len(self.images) == 0:
            logger.warning(f"No images found in {self.root_dir}")
    
    def _load_images(self):
        """이미지 로드"""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.glob('*.jpg'):
                try:
                    # 이미지 유효성 확인
                    with Image.open(img_path) as img:
                        img.verify()
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
                except Exception as e:
                    logger.warning(f"Skipping invalid image {img_path}: {e}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # 기본 변환
                image = transforms.ToTensor()(image)
            
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'path': img_path
            }
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # 대체 이미지 반환 (검은색)
            return {
                'image': torch.zeros(3, 224, 224),
                'label': torch.tensor(label, dtype=torch.long),
                'path': img_path
            }


def get_transforms(phase='train', image_size=224):
    """데이터 증강 변환 반환"""
    
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_dir, batch_size=64, num_workers=4, train_split=0.8):
    """DataLoader 생성"""
    
    # 데이터셋 로드
    full_dataset = DogCatDataset(
        root_dir=data_dir,
        transform=get_transforms('train')
    )
    
    # Train/Val 분할
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Val 데이터셋에 변환 적용
    val_dataset.dataset.transform = get_transforms('val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.idx_to_class
