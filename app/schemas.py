from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """예측 요청"""
    image_base64: str = Field(..., description="Base64 인코딩된 이미지")
    model_type: Optional[str] = "pytorch"
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANS...",
                "model_type": "pytorch"
            }
        }


class PredictionResponse(BaseModel):
    """예측 응답"""
    label: str = Field(..., description="예측 레이블 (dog, cat, unknown)")
    confidence: float = Field(..., description="신뢰도 (0-1)")
    status: str = Field(..., description="상태 (success, error, unknown)")
    error_message: Optional[str] = None
    processing_time_ms: float = Field(..., description="처리 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "dog",
                "confidence": 0.95,
                "status": "success",
                "processing_time_ms": 45.2
            }
        }


class ExplainabilityResponse(BaseModel):
    """설명 가능성 응답 (Grad-CAM)"""
    label: str
    confidence: float
    gradcam_image_base64: str
    original_image_base64: str
    overlay_image_base64: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "cat",
                "confidence": 0.92,
                "gradcam_image_base64": "iVBORw0KGgo...",
                "original_image_base64": "iVBORw0KGgo...",
                "overlay_image_base64": "iVBORw0KGgo..."
            }
        }


class ValidationResponse(BaseModel):
    """검증 응답"""
    is_valid: bool
    message: str
    reason: Optional[str]
    object_count: Optional[int]


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    version: str
    device: str


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    model_name: str
    backbone: str
    num_classes: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    parameters: int
