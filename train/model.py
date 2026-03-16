import torch
import torch.nn as nn
import timm
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DogCatClassifier(nn.Module):
    """Dog vs Cat 분류기"""
    
    def __init__(self, backbone='efficientnet_b0', num_classes=2, pretrained=True, dropout_rate=0.5):
        """
        Args:
            backbone: 백본 모델명
            num_classes: 출력 클래스 수
            pretrained: 사전학습 모델 사용 여부
            dropout_rate: Dropout 비율
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # 백본 모델 로드
        if backbone == 'resnet18':
            self.backbone = timm.create_model('resnet18', pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
            feature_dim = 1280
        elif backbone == 'convnext_tiny':
            self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained)
            feature_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # 분류 헤드
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        logger.info(f"Model initialized: {backbone} with {num_classes} classes")
    
    def forward(self, x):
        """전방향 전파"""
        features = self.backbone.forward_features(x)
        
        # 전역 평균 풀링
        if isinstance(features, torch.Tensor):
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
        
        out = self.classifier(features)
        return out
    
    def extract_features(self, x):
        """특징 추출"""
        return self.backbone.forward_features(x)


def create_model(config: Dict, device='cuda'):
    """설정에서 모델 생성"""
    
    model_config = config.get('model', {})
    model = DogCatClassifier(
        backbone=model_config.get('backbone', 'efficientnet_b0'),
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        dropout_rate=model_config.get('dropout_rate', 0.5)
    )
    
    model = model.to(device)
    return model


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
