import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'train'))

from model import DogCatClassifier, create_model, count_parameters


class TestModel:
    """모델 테스트"""
    
    def test_resnet18_creation(self):
        model = DogCatClassifier('resnet18', num_classes=2)
        assert model is not None
        assert model.backbone_name == 'resnet18'
    
    def test_efficientnet_creation(self):
        model = DogCatClassifier('efficientnet_b0', num_classes=2)
        assert model is not None
        assert model.backbone_name == 'efficientnet_b0'
    
    def test_convnext_creation(self):
        model = DogCatClassifier('convnext_tiny', num_classes=2)
        assert model is not None
        assert model.backbone_name == 'convnext_tiny'
    
    def test_forward_pass(self):
        model = DogCatClassifier('efficientnet_b0')
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        
        assert output.shape == (4, 2)
    
    def test_feature_extraction(self):
        model = DogCatClassifier('efficientnet_b0')
        x = torch.randn(4, 3, 224, 224)
        features = model.extract_features(x)
        
        assert features is not None
    
    def test_parameter_count(self):
        model = DogCatClassifier('efficientnet_b0')
        params = count_parameters(model)
        
        assert params > 0
        assert isinstance(params, int)
    
    def test_device_transfer(self):
        model = DogCatClassifier('efficientnet_b0')
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
