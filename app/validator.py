import torch
import torchvision.models.detection as detection_models
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageValidator:
    """이미지 검증 클래스"""
    
    def __init__(self, device='cuda', confidence_threshold=0.5):
        """
        Args:
            device: cuda 또는 cpu
            confidence_threshold: 객체 감지 신뢰도 임계값
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        
        # YOLO 대신 Faster R-CNN 사용 (torchvision 내장)
        try:
            self.detector = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
            self.detector.to(self.device)
            self.detector.eval()
            logger.info("Object detector loaded successfully")
        except Exception as e:
            logger.error("Failed to load detector: %s", e)
            self.detector = None
    
    def detect_objects(self, image):
        """이미지에서 객체 감지"""
        
        if self.detector is None:
            logger.warning("Detector not available, skipping object detection")
            return 0, []
        
        try:
            # 이미지 전처리
            image_tensor = self._preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            # 객체 감지
            with torch.no_grad():
                predictions = self.detector([image_tensor])
            
            # 신뢰도 필터링
            prediction = predictions[0]
            scores = prediction['scores'].cpu().numpy()
            boxes = prediction['boxes'].cpu().numpy()
            
            # 신뢰도가 높은 객체만 선택
            mask = scores > self.confidence_threshold
            filtered_boxes = boxes[mask]
            
            # 동물 클래스만 필터링 (dog: 18, cat: 17 in COCO)
            labels = prediction['labels'].cpu().numpy()
            animal_ids = [17, 18]  # COCO 클래스
            animal_mask = np.isin(labels, animal_ids)
            animal_mask = animal_mask & mask
            
            num_animals = np.sum(animal_mask)
            animal_boxes = boxes[animal_mask]
            
            return num_animals, animal_boxes
        
        except Exception as e:
            logger.error("Object detection error: %s", e)
            return 0, []
    
    def _preprocess_image(self, image):
        """이미지 전처리"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 값 정규화 (0-1)
        image = image.astype(np.float32) / 255.0
        
        # Tensor로 변환
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def check_image_validity(self, image):
        """이미지 유효성 검증"""
        
        results = {
            'is_valid': True,
            'message': '검증 성공',
            'issues': []
        }
        
        try:
            # 이미지 크기 확인
            if image.size[0] < 50 or image.size[1] < 50:
                results['is_valid'] = False
                results['issues'].append('이미지가 너무 작습니다')
            
            # 채널 확인
            if image.mode != 'RGB':
                try:
                    image = image.convert('RGB')
                except:
                    results['is_valid'] = False
                    results['issues'].append('이미지 형식이 지원되지 않습니다')
            
            # 객체 수 확인
            num_objects, _ = self.detect_objects(image)
            if num_objects > 1:
                results['is_valid'] = False
                results['issues'].append(f'여러 동물이 감지되었습니다 ({num_objects}개)')
                results['object_count'] = num_objects
            elif num_objects == 0:
                results['is_valid'] = False
                results['issues'].append('이미지에 동물이 감지되지 않았습니다')
        
        except Exception as e:
            logger.error("Validation error: %s", e)
            results['is_valid'] = False
            results['issues'].append(str(e))
        
        return results


class ConfidenceThresholdTuner:
    """신뢰도 임계값 조정 클래스"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def entropy(probabilities):
        """엔트로피 계산"""
        eps = 1e-8
        return -np.sum(probabilities * np.log(probabilities + eps))
    
    @staticmethod
    def margin(probabilities):
        """마진 계산"""
        sorted_probs = np.sort(probabilities)[::-1]
        return sorted_probs[0] - sorted_probs[1]
    
    @staticmethod
    def is_unknown(probabilities, confidence_threshold=0.7, entropy_threshold=0.5):
        """미지 샘플 판정"""
        max_prob = np.max(probabilities)
        ent = ConfidenceThresholdTuner.entropy(probabilities)
        margin = ConfidenceThresholdTuner.margin(probabilities)
        
        # 신뢰도 낮음
        if max_prob < confidence_threshold:
            return True, 'low_confidence'
        
        # 엔트로피 높음
        if ent > entropy_threshold:
            return True, 'high_entropy'
        
        # 마진 작음
        if margin < 0.1:
            return True, 'low_margin'
        
        return False, 'known'
