# ============================================================================
# analysis/services/prediction_service.py
# Plant Disease Prediction Service — PyTorch, Lazy Loading, Thresholds
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "model_files", "efficientnet_b0_plant_disease.pth")
_CLASS_NAMES_PATH = os.path.join(_BASE_DIR, "model_files", "class_names.json")

# Confidence threshold — predictions below this are flagged as "low confidence"
CONFIDENCE_THRESHOLD = 40.0  # percent


# ============================================================================
# Prediction Result
# ============================================================================

@dataclass
class PredictionResult:
    """Structured prediction result returned by the classifier."""
    disease: str           # Display-ready disease name
    confidence: float      # 0-100 numeric (NOT a string)
    is_healthy: bool
    is_low_confidence: bool
    recommendations: List[str]
    raw_class: str         # Original class label from class_names.json
    class_index: int       # Predicted class index
    probabilities: List[float]
    inference_time_ms: float
    top_k: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease": self.disease,
            "confidence": round(self.confidence, 2),
            "is_healthy": self.is_healthy,
            "is_low_confidence": self.is_low_confidence,
            "recommendations": self.recommendations,
            "raw_class": self.raw_class,
            "class_index": self.class_index,
            "probabilities": self.probabilities,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "top_k": self.top_k,
        }


# ============================================================================
# Display Name Formatting
# ============================================================================

def _format_display_name(raw_class: str) -> str:
    """
    Convert raw class label to a clean display name.

    Examples:
        'Tomato__early_blight'  -> 'Tomato Early Blight'
        'Apple__black_rot'      -> 'Apple Black Rot'
        'Mulberry Leaf Rust'    -> 'Mulberry Leaf Rust'
        'olivepeacockspot'      -> 'Olive Peacock Spot'
    """
    # Replace separators with spaces
    name = raw_class.replace("__", " ").replace("_", " ")
    # Title case
    name = name.title()
    # Remove duplicate consecutive words (e.g., "Tomato Tomato" -> "Tomato")
    words = name.split()
    unique_words = []
    for w in words:
        if not unique_words or unique_words[-1].lower() != w.lower():
            unique_words.append(w)
    return " ".join(unique_words)


def _normalize_lookup_key(raw_class: str) -> str:
    """
    Normalize a class label for treatment lookup.

    Converts to lowercase, replaces spaces with underscores,
    strips leading/trailing underscores.

    Examples:
        'Tomato__early_blight' -> 'tomato__early_blight'
        'Mulberry Leaf Rust'   -> 'mulberry_leaf_rust'
    """
    return raw_class.lower().strip().replace(" ", "_")


# ============================================================================
# Model Architecture Factory
# ============================================================================

class PlantDiseaseModel:
    """Factory for the EfficientNet-B0 architecture."""

    @staticmethod
    def build(num_classes: int):
        import torch.nn as nn
        from torchvision import models

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model


# ============================================================================
# Disease Classifier
# ============================================================================

class DiseaseClassifier:
    """
    PyTorch-based plant disease classifier.

    Features:
        - Lazy model loading (loaded on first predict call, not at import)
        - Thread-safe singleton pattern
        - Confidence thresholds with low-confidence warnings
        - Prediction timing for performance monitoring
        - Top-K predictions for uncertainty analysis
    """

    _instance: Optional["DiseaseClassifier"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DiseaseClassifier":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model = None
        self._reverse_class_indices: Optional[Dict[int, str]] = None
        self._model_loaded = False
        self._initialized = True

    def _ensure_model_loaded(self) -> None:
        """Lazily load the model on first use."""
        if self._model_loaded:
            return

        with self._lock:
            if self._model_loaded:
                return

            logger.info("Loading PyTorch model (lazy)...")
            start = time.time()

            import torch
            import torch.nn as nn

            # Load class names
            with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                class_names = json.load(f)
            self._reverse_class_indices = {int(k): v for k, v in class_names.items()}
            num_classes = len(self._reverse_class_indices)

            # Build model architecture
            self._model = PlantDiseaseModel.build(num_classes)

            # Load trained weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(_MODEL_PATH, map_location=device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.to(device)
            self._model.eval()

            self._device = device

            elapsed = (time.time() - start) * 1000
            logger.info(
                "Model loaded in %.0f ms: %d classes, device=%s",
                elapsed,
                num_classes,
                device,
            )
            self._model_loaded = True

    @property
    def model(self):
        """Access the loaded PyTorch model."""
        self._ensure_model_loaded()
        return self._model

    @property
    def reverse_class_indices(self) -> Dict[int, str]:
        """Access the class index mapping."""
        self._ensure_model_loaded()
        return self._reverse_class_indices

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    def predict(self, image_file) -> PredictionResult:
        """
        Predict disease from an uploaded image.

        Args:
            image_file: File-like object (seekable)

        Returns:
            PredictionResult with disease, confidence, recommendations, etc.
        """
        import torch

        from .preprocessing import preprocess_for_model
        from .treatment_service import find_recommendations, DEFAULT_HEALTHY_PRACTICES
        from .logging_service import performance_tracker

        self._ensure_model_loaded()

        # Preprocess
        img_tensor, _ = preprocess_for_model(image_file)
        img_tensor = img_tensor.to(self._device)

        # Inference with timing
        t0 = time.time()
        with torch.no_grad():
            outputs = self._model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predictions = probabilities[0].cpu().numpy()
        inference_ms = (time.time() - t0) * 1000

        # Track performance
        performance_tracker.record_inference(inference_ms)

        # Extract results
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx]) * 100

        pred_class = self._reverse_class_indices.get(predicted_idx, "Unknown")
        lookup_key = _normalize_lookup_key(pred_class)
        display_name = _format_display_name(pred_class)

        # Determine recommendations
        recommendations = find_recommendations(lookup_key, pred_class)
        is_healthy = "healthy" in lookup_key.lower()
        if is_healthy:
            recommendations = list(DEFAULT_HEALTHY_PRACTICES)

        # Confidence threshold check
        is_low_confidence = confidence < CONFIDENCE_THRESHOLD

        # Top-K predictions for uncertainty analysis
        top_k_indices = np.argsort(predictions)[-5:][::-1]
        top_k = [
            {
                "class": self._reverse_class_indices.get(int(idx), "Unknown"),
                "display_name": _format_display_name(
                    self._reverse_class_indices.get(int(idx), "Unknown")
                ),
                "confidence": round(float(predictions[idx]) * 100, 2),
            }
            for idx in top_k_indices
        ]

        result = PredictionResult(
            disease=display_name,
            confidence=confidence,
            is_healthy=is_healthy,
            is_low_confidence=is_low_confidence,
            recommendations=recommendations,
            raw_class=pred_class,
            class_index=predicted_idx,
            probabilities=predictions.tolist(),
            inference_time_ms=inference_ms,
            top_k=top_k,
        )

        logger.info(
            "Prediction: %s (%.2f%%) in %.0f ms%s",
            result.disease,
            result.confidence,
            inference_ms,
            " [LOW CONFIDENCE]" if is_low_confidence else "",
        )

        return result


# ============================================================================
# Module-level accessor
# ============================================================================

def get_classifier() -> DiseaseClassifier:
    """Get or create the singleton classifier instance."""
    return DiseaseClassifier()
