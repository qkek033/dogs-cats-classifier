"""
Grad-CAM 설명 가능성 시각화.
- 마지막 Conv2d 레이어 자동 탐지
- register_full_backward_hook 사용 (deprecated register_backward_hook 대체)
- GPU/CPU 일관된 device 처리
- cam, heatmap, overlay 반환 (Streamlit 호환)
"""
import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


def get_last_conv_layer(model: nn.Module) -> Optional[Tuple[str, nn.Module]]:
    """
    모델에서 마지막 Conv2d 레이어를 찾습니다.
    ResNet: layer4[-1], EfficientNet: conv_head, ConvNeXt: features[-1] 등.
    """
    last_conv: Optional[Tuple[str, nn.Module]] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)
    return last_conv


class GradCAM:
    """Grad-CAM 시각화 (단일 타깃 레이어, 최신 훅 사용)."""

    def __init__(self, model: nn.Module, target_layer: Optional[Tuple[str, nn.Module]] = None):
        """
        Args:
            model: PyTorch 모델 (eval 모드로 사용).
            target_layer: (name, module) 또는 None이면 마지막 Conv2d 자동 선택.
        """
        self.model = model
        self._target_name: Optional[str] = None
        self._target_module: Optional[nn.Module] = None
        if target_layer is not None:
            self._target_name, self._target_module = target_layer
        else:
            found = get_last_conv_layer(model)
            if found is None:
                raise ValueError("No Conv2d layer found in model for Grad-CAM")
            self._target_name, self._target_module = found

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._forward_handle: Optional[Any] = None
        self._backward_handle: Optional[Any] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """순전파/역전파 훅 등록 (register_full_backward_hook 사용)."""

        def forward_hook(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            self._activations = output.detach()

        def backward_hook(
            _module: nn.Module,
            _grad_input: Tuple[torch.Tensor, ...],
            grad_output: Tuple[torch.Tensor, ...],
        ) -> None:
            self._gradients = grad_output[0].detach()

        self._forward_handle = self._target_module.register_forward_hook(forward_hook)
        self._backward_handle = self._target_module.register_full_backward_hook(backward_hook)
        logger.debug("Grad-CAM target layer: %s", self._target_name)

    def remove_hooks(self) -> None:
        """훅 제거."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None
        self._activations = None
        self._gradients = None
        logger.debug("Grad-CAM hooks removed")

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        device: torch.device,
    ) -> np.ndarray:
        """
        Grad-CAM 맵 생성.
        weights = global_average_pooling(gradients)
        cam = ReLU(weighted_sum(weights * activations)), then normalize.
        """
        self.model.eval()
        self._activations = None
        self._gradients = None

        input_tensor = input_tensor.to(device)
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(input_tensor)
        target_score = output[0, class_idx]
        target_score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Grad-CAM: gradients or activations were not captured. Check target layer.")

        # 같은 device에서 처리 후 numpy로 변환
        grads = self._gradients
        acts = self._activations
        logger.debug(
            "Grad-CAM activation shape: %s, gradient shape: %s",
            tuple(acts.shape),
            tuple(grads.shape),
        )

        # weights = global average pooling over spatial dimensions
        weights = grads.mean(dim=(2, 3))
        # cam = weighted sum over channels
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(1)
        cam_np = cam.detach().cpu().numpy()[0]
        # normalize to [0, 1]
        cam_max = cam_np.max()
        if cam_max > 1e-8:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)
        logger.debug("Grad-CAM generation success, cam shape: %s", cam_np.shape)
        return cam_np.astype(np.float32)

    def visualize(
        self,
        image_tensor: torch.Tensor,
        original_image: Image.Image,
        class_idx: int,
        class_name: str,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """
        Grad-CAM 시각화: cam, heatmap, overlay (RGB uint8) 반환.
        Streamlit st.image()에 사용 가능하도록 numpy array 반환.
        """
        cam = self.generate_cam(image_tensor, class_idx, device)
        h, w = original_image.size[1], original_image.size[0]
        cam_resized = cv2.resize(cam, (w, h))
        cam_resized = np.clip(cam_resized, 0.0, 1.0)

        heatmap = cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        original_np = np.array(original_image)
        if original_np.ndim == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
        elif original_np.shape[2] == 4:
            original_np = original_np[:, :, :3]
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return {
            "cam": cam_resized,
            "heatmap": heatmap,
            "overlay": overlay,
        }
