# ============================================================================
# analysis/services/gradcam_service.py
# Grad-CAM Explainability Service — Heatmap Generation & Disease Progress
# ============================================================================

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# Grad-CAM Generation
# ============================================================================

def _find_last_conv_layer(model) -> Optional[Tuple[int, str]]:
    """
    Find the last Conv2D layer index and name in a Keras model.

    Returns:
        (index, name) tuple or None if no Conv2D found
    """
    import tensorflow as tf

    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return i, model.layers[i].name
    return None


def get_conv_layer_names(model) -> List[str]:
    """
    Return all Conv2D layer names for debugging/layer selection.

    Args:
        model: Loaded Keras model

    Returns:
        List of Conv2D layer names
    """
    import tensorflow as tf

    return [
        layer.name
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]


def generate_gradcam(
    model,
    image_file,
    target_layer_idx: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate Grad-CAM heatmap for a TensorFlow/Keras model.

    Uses manual forward pass through layers for compatibility with
    any Keras Sequential model. Records gradients via GradientTape
    with explicit tape.watch() for reliable gradient computation.

    Args:
        model: Loaded Keras model
        image_file: File-like object (seekable) — the uploaded image
        target_layer_idx: Optional specific layer index to target.
                         If None, uses the last Conv2D layer.

    Returns:
        dict with:
            - original: PIL Image (original size)
            - heatmap: PIL Image (colorized heatmap, original size)
            - superimposed: PIL Image (blended overlay, original size)
            - predicted_class_idx: int
            - target_layer: str (layer name used)
            - activation_stats: dict with mean/max/min/std/coverage
            - gradcam_time_ms: float
        or None on failure
    """
    import tensorflow as tf
    from .preprocessing import preprocess_for_model
    from .logging_service import performance_tracker

    t0 = time.time()

    try:
        # Load original image at full resolution
        image_file.seek(0)
        original_img = Image.open(image_file).convert("RGB")
        original_np = np.array(original_img)

        # Preprocess for model
        image_file.seek(0)
        img_array, _ = preprocess_for_model(image_file)

        # Find target layer
        if target_layer_idx is not None:
            target_name = model.layers[target_layer_idx].name
        else:
            result = _find_last_conv_layer(model)
            if result is None:
                logger.warning("No Conv2D layer found for Grad-CAM")
                return None
            target_layer_idx, target_name = result

        # Convert to tensor and watch for gradients
        img_tensor = tf.constant(img_array, dtype=tf.float32)

        # Manual forward pass to capture intermediate conv output
        with tf.GradientTape() as tape:
            conv_output = None
            x = img_tensor
            for idx, layer in enumerate(model.layers):
                x = layer(x, training=False)
                if idx == target_layer_idx:
                    conv_output = x
                    tape.watch(conv_output)  # Explicitly watch conv output
            predictions = x

            # Target: predicted class score
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Compute gradients of predicted class w.r.t. conv output
        grads = tape.gradient(loss, conv_output)
        if grads is None:
            logger.warning("Could not compute gradients for Grad-CAM")
            return None

        # Global average pooling of gradients → channel weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weighted combination of feature maps
        conv_squeezed = conv_output[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_squeezed), axis=-1
        )

        # ReLU + normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        heatmap = heatmap / (max_val + 1e-8)
        heatmap_np = heatmap.numpy()

        # Resize heatmap to original image dimensions
        heatmap_resized = cv2.resize(
            heatmap_np,
            (original_np.shape[1], original_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap and create superimposed image
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(
            original_bgr, 0.6, heatmap_colored, 0.4, 0
        )

        # Compute activation statistics
        activation_stats = {
            "mean": float(np.mean(heatmap_uint8)),
            "max": float(np.max(heatmap_uint8)),
            "min": float(np.min(heatmap_uint8)),
            "std": float(np.std(heatmap_uint8)),
            "coverage_percent": float(
                np.sum(heatmap_uint8 > 100) / heatmap_uint8.size * 100
            ),
        }

        elapsed_ms = (time.time() - t0) * 1000
        performance_tracker.record_gradcam(elapsed_ms)

        return {
            "original": original_img,
            "heatmap": Image.fromarray(
                cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            ),
            "superimposed": Image.fromarray(
                cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            ),
            "predicted_class_idx": int(class_idx.numpy()),
            "target_layer": target_name,
            "activation_stats": activation_stats,
            "gradcam_time_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        logger.error(
            "Grad-CAM Error: %s: %s", type(e).__name__, e, exc_info=True
        )
        return None


# ============================================================================
# Disease Progress Estimation
# ============================================================================

def estimate_disease_progress(
    gradcam_pil: Optional[Image.Image],
) -> Tuple[float, str]:
    """
    Estimate disease spread ratio based on Grad-CAM heatmap colors.

    Analyzes the HSV color distribution in the heatmap to estimate
    how much of the leaf area shows infection markers.

    Args:
        gradcam_pil: PIL Image of Grad-CAM superimposed visualization

    Returns:
        (ratio, stage): infection percentage and severity stage string
    """
    if gradcam_pil is None:
        return 0.0, "Unknown"

    img = np.array(gradcam_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Red regions (severe infection — high activation)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        + cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    )

    # Yellow regions (moderate infection)
    yellow_mask = cv2.inRange(
        hsv, np.array([20, 120, 120]), np.array([35, 255, 255])
    )

    # Green regions (healthy tissue)
    green_mask = cv2.inRange(
        hsv, np.array([40, 40, 40]), np.array([85, 255, 255])
    )

    red_pixels = int(np.sum(red_mask > 0))
    yellow_pixels = int(np.sum(yellow_mask > 0))
    green_pixels = int(np.sum(green_mask > 0))

    infected = red_pixels + yellow_pixels
    leaf_area = infected + green_pixels

    ratio = (infected / leaf_area * 100) if leaf_area > 0 else 0.0

    if ratio < 5:
        stage = "Very Early"
    elif ratio < 20:
        stage = "Early"
    elif ratio < 40:
        stage = "Moderate"
    elif ratio < 70:
        stage = "Advanced"
    else:
        stage = "Severe"

    return round(ratio, 2), stage


# ============================================================================
# Bounding Box Visualization
# ============================================================================

def draw_red_regions_boxes(
    gradcam_pil: Optional[Image.Image],
) -> Optional[Image.Image]:
    """
    Draw bounding boxes around red (high-activation) regions in the heatmap.

    Args:
        gradcam_pil: PIL Image of Grad-CAM superimposed visualization

    Returns:
        PIL Image with bounding boxes drawn, or None
    """
    if gradcam_pil is None:
        return None

    img = np.array(gradcam_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Detect red regions (high activation)
    mask = (
        cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        + cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    )

    # Morphological closing to merge nearby regions
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return Image.fromarray(img)
