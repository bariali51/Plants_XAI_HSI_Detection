# ============================================================================
# analysis/services/preprocessing.py
# Image Preprocessing for PyTorch Plant Disease Model
# ============================================================================

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Model expects 224x224 input
IMG_SIZE: Tuple[int, int] = (224, 224)

# Standard ImageNet normalization used during training
_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_for_model(image_file) -> Tuple[torch.Tensor, Image.Image]:
    """
    Prepare image for PyTorch model inference.

    Args:
        image_file: File-like object (seekable)

    Returns:
        img_tensor: Preprocessed tensor with batch dimension [1, 3, 256, 256]
        img: Original PIL Image (resized)
    """
    image_file.seek(0)
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize(IMG_SIZE)

    img_tensor = _transform(img_resized).unsqueeze(0)  # Add batch dim
    return img_tensor, img_resized