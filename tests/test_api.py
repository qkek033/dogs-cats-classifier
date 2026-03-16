import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from main import app

client = TestClient(app)


def create_test_image(size=(224, 224)):
    """Create a test RGB image."""
    img = Image.new('RGB', size, color=(73, 109, 137))
    return img


def encode_image_to_base64(image):
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


class TestHealthCheck:
    """Health check endpoint tests."""

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestPrediction:
    """Prediction endpoint tests."""

    def test_predict_with_valid_image(self):
        """Predict with valid image."""
        image = create_test_image()
        image_b64 = encode_image_to_base64(image)
        
        response = client.post(
            "/predict",
            json={"image_base64": image_b64}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert data["status"] in ["success", "unknown_detected", "validation_failed"]
    
    def test_predict_invalid_base64(self):
        """Predict with invalid base64 returns 4xx/5xx."""
        response = client.post(
            "/predict",
            json={"image_base64": "invalid_base64"}
        )
        
        assert response.status_code in [400, 500]


class TestValidation:
    """Validation endpoint tests."""

    def test_validate_image(self):
        """Validate image upload."""
        image = create_test_image()
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        response = client.post(
            "/validate",
            files={"file": ("test.png", img_byte_arr, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "message" in data


class TestModelInfo:
    """Model info endpoint tests."""

    def test_get_model_info(self):
        response = client.get("/model-info")
        
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "backbone" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
