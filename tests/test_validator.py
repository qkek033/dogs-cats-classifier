import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from validator import ImageValidator, ConfidenceThresholdTuner


class TestImageValidator:
    """이미지 검증기 테스트"""
    
    @pytest.fixture
    def validator(self):
        return ImageValidator(device='cpu')
    
    def test_valid_image(self, validator):
        """유효한 이미지 검증"""
        image = Image.new('RGB', (224, 224))
        result = validator.check_image_validity(image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
    
    def test_small_image(self, validator):
        """작은 이미지 검증"""
        image = Image.new('RGB', (30, 30))
        result = validator.check_image_validity(image)
        
        assert result['is_valid'] is False
    
    def test_image_conversion(self, validator):
        """이미지 포맷 변환"""
        image = Image.new('RGBA', (224, 224))
        result = validator.check_image_validity(image)
        
        # RGB로 변환되어야 함
        assert isinstance(result, dict)


class TestConfidenceThreshold:
    """신뢰도 임계값 테스트"""
    
    def test_entropy_calculation(self):
        """엔트로피 계산"""
        probs = np.array([0.9, 0.1])
        entropy = ConfidenceThresholdTuner.entropy(probs)
        
        assert entropy >= 0
    
    def test_margin_calculation(self):
        """마진 계산"""
        probs = np.array([0.8, 0.2])
        margin = ConfidenceThresholdTuner.margin(probs)
        
        assert margin == 0.6
    
    def test_unknown_detection_high_confidence(self):
        """높은 신뢰도"""
        probs = np.array([0.95, 0.05])
        is_unknown, reason = ConfidenceThresholdTuner.is_unknown(probs)
        
        assert is_unknown is False
    
    def test_unknown_detection_low_confidence(self):
        """낮은 신뢰도"""
        probs = np.array([0.55, 0.45])
        is_unknown, reason = ConfidenceThresholdTuner.is_unknown(probs)
        
        # 낮은 신뢰도이므로 unknown으로 판정
        assert is_unknown is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
