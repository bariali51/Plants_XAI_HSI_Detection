# ============================================================================
# analysis/ml_utils.py
# Plant Disease Detection with XAI (Grad-CAM)
# TensorFlow/Keras Implementation — Optimized
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = [
    "classifier_instance",
    "estimate_disease_progress",
    "draw_red_regions_boxes",
    "ai_treatment_advisor",
    "ai_llm_doctor",
    "ai_gemini_doctor",
    "ai_compare_evolution",
    "ai_doctor_report",
    "get_treatment",
    "apply_gradcam_standalone",
    "generate_gradcam_visualizations",
    "export_gradcam_report",
]

# ============================================================================
# File Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_files", "plant_disease_prediction_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "model_files", "class_indices.json")
TREATMENTS_PATH = os.path.join(BASE_DIR, "model_files", "treatments.json")

# Model expects 128x128 input (verified from model.summary())
IMG_SIZE: Tuple[int, int] = (128, 128)

# ============================================================================
# Treatment Recommendations
# ============================================================================

with open(TREATMENTS_PATH, "r", encoding="utf-8") as _f:
    _treatment_data = json.load(_f)

TREATMENT_RECOMMENDATIONS: Dict[str, List[str]] = _treatment_data["TREATMENT_RECOMMENDATIONS"]
DEFAULT_HEALTHY_PRACTICES: List[str] = _treatment_data["DEFAULT_HEALTHY_PRACTICES"]


def get_treatment(predicted_label: str) -> List[str]:
    """Get treatment recommendations for a predicted disease label."""
    return TREATMENT_RECOMMENDATIONS.get(predicted_label, DEFAULT_HEALTHY_PRACTICES)


# ============================================================================
# Disease Classifier
# ============================================================================

class DiseaseClassifier:
    """TensorFlow-based plant disease classifier with Grad-CAM explainability."""

    def __init__(self) -> None:
        self.model: Optional[tf.keras.Model] = None
        self.reverse_class_indices: Optional[Dict[int, str]] = None
        self.load_resources()

    def load_resources(self) -> None:
        """Load TensorFlow model and class indices."""
        logger.info("Loading TensorFlow model...")

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        self.reverse_class_indices = {int(k): v for k, v in class_indices.items()}

        logger.info(
            "Model loaded: %d classes, input shape %s",
            len(self.reverse_class_indices),
            self.model.input_shape,
        )

    def preprocess_image(self, image_file) -> Tuple[np.ndarray, Image.Image]:
        """
        Prepare image for Keras model inference.

        Args:
            image_file: File-like object (seekable)

        Returns:
            img_array: Preprocessed numpy array with batch dimension
            img: Original PIL Image
        """
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
        img = img.resize(IMG_SIZE)

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img

    def predict(self, image_file) -> Dict[str, Any]:
        """
        Predict disease from uploaded image.

        Args:
            image_file: File-like object (seekable)

        Returns:
            dict with disease, confidence, recommendations, etc.
        """
        img_array, _ = self.preprocess_image(image_file)

        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx]) * 100

        pred_class = self.reverse_class_indices.get(predicted_idx, "Unknown")
        lookup_key = pred_class.lower().strip().replace(" ", "_")

        recommendations = self._find_recommendations(lookup_key, pred_class)

        if "healthy" in lookup_key.lower():
            recommendations = DEFAULT_HEALTHY_PRACTICES

        return {
            "disease": pred_class.replace("_", " ").title(),
            "confidence": f"{confidence:.2f}%",
            "recommendations": recommendations,
            "is_healthy": "healthy" in lookup_key.lower(),
            "raw_class": pred_class,
            "class_index": predicted_idx,
            "probabilities": predictions.tolist(),
        }

    def _find_recommendations(self, lookup_key: str, raw_class: str) -> List[str]:
        """Flexible recommendation lookup with partial matching."""
        # Direct match
        if lookup_key in TREATMENT_RECOMMENDATIONS:
            return TREATMENT_RECOMMENDATIONS[lookup_key]

        # Partial match
        for key in TREATMENT_RECOMMENDATIONS:
            if key in lookup_key or lookup_key in key:
                return TREATMENT_RECOMMENDATIONS[key]

        # Fallback: match plant type
        plant = raw_class.split("__")[0].lower() if "__" in raw_class else raw_class.lower()
        for key in TREATMENT_RECOMMENDATIONS:
            if plant in key:
                return TREATMENT_RECOMMENDATIONS[key]

        return ["Consult an agricultural specialist for specific treatment."]

    def _get_last_conv_layer_index(self) -> Optional[Tuple[int, str]]:
        """Find the last Conv2D layer index and name."""
        for i in range(len(self.model.layers) - 1, -1, -1):
            if isinstance(self.model.layers[i], tf.keras.layers.Conv2D):
                return i, self.model.layers[i].name
        return None

    def apply_gradcam(self, image_file, layer_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate Grad-CAM heatmap for TensorFlow/Keras models.
        Uses manual forward pass for compatibility with any Keras Sequential model.

        Args:
            image_file: File-like object (seekable)
            layer_name: Optional specific layer name

        Returns:
            dict with original, heatmap, superimposed PIL Images + metadata
        """
        try:
            image_file.seek(0)
            original_img = Image.open(image_file).convert("RGB")
            original_np = np.array(original_img)

            image_file.seek(0)
            img_array, _ = self.preprocess_image(image_file)

            result = self._get_last_conv_layer_index()
            if result is None:
                logger.warning("No Conv2D layer found for Grad-CAM")
                return None
            target_layer_idx, target_layer_name = result

            def forward_with_intermediate(x):
                output = x
                conv_output = None
                for idx, layer in enumerate(self.model.layers):
                    output = layer(output, training=False)
                    if idx == target_layer_idx:
                        conv_output = output
                return conv_output, output

            with tf.GradientTape() as tape:
                conv_features, predictions = forward_with_intermediate(img_array)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_features)
            if grads is None:
                logger.warning("Could not compute gradients for Grad-CAM")
                return None

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_features_squeezed = conv_features[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_features_squeezed), axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

            heatmap_np = heatmap.numpy()
            heatmap_np = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_np)

            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
            superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

            pred_idx = int(np.argmax(predictions[0]))
            pred_class = self.reverse_class_indices.get(pred_idx, "Unknown")

            return {
                "original": original_img,
                "heatmap": Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)),
                "superimposed": Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)),
                "predicted_class": pred_class,
                "target_layer": target_layer_name,
                "activation_stats": {
                    "mean": float(np.mean(heatmap_uint8)),
                    "max": float(np.max(heatmap_uint8)),
                    "min": float(np.min(heatmap_uint8)),
                    "std": float(np.std(heatmap_uint8)),
                    "coverage_percent": float(np.sum(heatmap_uint8 > 100) / heatmap_uint8.size * 100),
                },
            }

        except Exception as e:
            logger.error("Grad-CAM Error: %s: %s", type(e).__name__, e, exc_info=True)
            return None


# Singleton instance
classifier_instance = DiseaseClassifier()


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_disease_progress(gradcam_pil: Optional[Image.Image]) -> Tuple[float, str]:
    """
    Estimate disease spread ratio based on Grad-CAM heatmap colors.

    Args:
        gradcam_pil: PIL Image of Grad-CAM heatmap

    Returns:
        (ratio, stage): infection percentage and severity stage
    """
    if gradcam_pil is None:
        return 0.0, "Unknown"

    img = np.array(gradcam_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Red regions (severe infection)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        + cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    )

    # Yellow regions (moderate infection)
    yellow_mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([35, 255, 255]))

    # Green regions (healthy)
    green_mask = cv2.inRange(hsv, np.array([40, 40, 40]), np.array([85, 255, 255]))

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


def draw_red_regions_boxes(gradcam_pil: Optional[Image.Image]) -> Optional[Image.Image]:
    """
    Draw bounding boxes around red (high-activation) regions in the heatmap.

    Args:
        gradcam_pil: PIL Image of Grad-CAM heatmap

    Returns:
        PIL Image with bounding boxes drawn
    """
    if gradcam_pil is None:
        return None

    img = np.array(gradcam_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = (
        cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        + cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    )

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return Image.fromarray(img)


def ai_treatment_advisor(
    disease_name: str, confidence: float, stage: str
) -> List[str]:
    """
    Smart treatment advisor based on disease type and severity stage.

    Args:
        disease_name: Predicted disease name
        confidence: Prediction confidence (0-100)
        stage: Disease severity stage

    Returns:
        list of treatment advice strings
    """
    disease_lower = disease_name.lower()
    advice: List[str] = []

    if "early_blight" in disease_lower:
        advice.extend([
            "Remove infected leaves immediately to prevent spread.",
            "Apply copper-based fungicide every 7 days.",
            "Avoid overhead irrigation to reduce humidity.",
        ])
    elif "late_blight" in disease_lower:
        advice.extend([
            "Isolate affected plants urgently.",
            "Use systemic fungicides containing metalaxyl.",
            "Improve field drainage and airflow.",
        ])
    elif "bacterial_spot" in disease_lower:
        advice.extend([
            "Use certified disease-free seeds.",
            "Spray copper bactericides weekly.",
            "Rotate crops next season.",
        ])
    elif "leaf_mold" in disease_lower:
        advice.extend([
            "Increase greenhouse ventilation.",
            "Reduce leaf wetness duration.",
            "Apply preventive fungicide.",
        ])
    elif "healthy" in disease_lower:
        advice.extend([
            "Plant appears healthy.",
            "Maintain balanced fertilization.",
            "Monitor regularly for early symptoms.",
        ])
    else:
        advice.extend([
            "Consult agricultural specialist.",
            "Monitor disease progression closely.",
            "Apply broad-spectrum fungicide if necessary.",
        ])

    if stage == "Advanced":
        advice.append("Disease is advanced - immediate chemical control recommended.")
    if confidence > 90:
        advice.append("AI confidence is high - treatment should start immediately.")

    return advice


def ai_llm_doctor(
    disease: str, confidence: float, stage: str, ratio: float
) -> str:
    """
    Medical consultation via OpenAI GPT.

    Args:
        disease: Predicted disease name
        confidence: Prediction confidence
        stage: Disease severity stage
        ratio: Infection ratio percentage

    Returns:
        AI-generated medical advice string
    """
    try:
        from openai import OpenAI
        from django.conf import settings

        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = f"""
You are an expert plant disease doctor.

Disease: {disease}
Confidence: {confidence}%
Stage: {stage}
Severity ratio: {ratio}%

Give professional treatment explanation and agronomic advice.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a plant pathology expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=250,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.warning("OpenAI API unavailable: %s", e)
        return (
            "AI Doctor service unavailable.\n\n"
            "Recommended immediate actions:\n"
            "- Monitor disease progression carefully.\n"
            "- Apply preventive fungicide treatment.\n"
            "- Improve irrigation balance and airflow.\n"
            "- Consult local agricultural expert.\n\n"
            "(System fallback activated)"
        )


def ai_gemini_doctor(
    disease: str, confidence: float, stage: str, ratio: float
) -> Dict[str, Any]:
    """
    Medical consultation via Google Gemini.

    Args:
        disease: Predicted disease name
        confidence: Prediction confidence
        stage: Disease severity stage
        ratio: Infection ratio percentage

    Returns:
        dict: Structured medical report
    """
    try:
        import google.generativeai as genai
        from django.conf import settings

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-flash-latest")

        prompt = f"""
You are a professional plant disease agronomist AI.

Give response STRICTLY in this format:

MEDICAL:
(short scientific diagnosis)

TREATMENT:
(clear fungicide and agronomic actions)

IRRIGATION:
(water management advice)

ECONOMIC:
(yield loss risk and financial impact)

Disease: {disease}
Confidence: {confidence}
Stage: {stage}
Infection ratio: {ratio}%
"""
        response = model.generate_content(prompt)
        text = response.text

        # Parse structured response
        sections = text.split("TREATMENT:")
        medical_text = sections[0].replace("MEDICAL:", "").strip() if sections else "N/A"

        rest = sections[1].split("IRRIGATION:") if len(sections) > 1 else ["", ""]
        treatment_text = rest[0].strip()

        rest2 = rest[1].split("ECONOMIC:") if len(rest) > 1 else ["", ""]
        irrigation_text = rest2[0].strip()
        risk_text = rest2[1].strip() if len(rest2) > 1 else "N/A"

        yield_loss = int(ratio * 0.6)

        return {
            "medical": medical_text,
            "treatment": treatment_text,
            "irrigation": irrigation_text,
            "economic_risk": risk_text,
            "yield_loss_percent": yield_loss,
            "fungicides": [
                {"name": "Mancozeb", "type": "Protectant"},
                {"name": "Difenoconazole", "type": "Systemic"},
            ],
        }

    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return {
            "medical": "AI doctor unavailable",
            "treatment": "-",
            "irrigation": "-",
            "economic_risk": "-",
            "yield_loss_percent": 0,
            "fungicides": [],
        }


def ai_compare_evolution(
    old_disease: str, old_ratio: float, new_ratio: float
) -> str:
    """
    Compare disease evolution between two scans.

    Args:
        old_disease: Previous disease prediction
        old_ratio: Previous infection ratio
        new_ratio: Current infection ratio

    Returns:
        Evolution analysis report string
    """
    diff = new_ratio - old_ratio

    if diff > 15:
        trend = "Disease has progressed aggressively."
        status = "Severe deterioration"
    elif diff > 5:
        trend = "Disease progression detected."
        status = "Condition worsening"
    elif diff > -5:
        trend = "Disease remains relatively stable."
        status = "Stable"
    else:
        trend = "Disease regression observed. Plant health improving."
        status = "Improvement"

    return f"""
AI Evolution Analysis:

Plant disease: {old_disease}

Previous infection level: {old_ratio:.2f}%
Current infection level: {new_ratio:.2f}%

Overall evolution status: {status}

Interpretation:
{trend}

Recommendation:
Continuous monitoring is strongly advised.
Adjust fungicide program according to disease dynamics.
Protect remaining healthy foliage to secure yield potential.
"""


def ai_doctor_report(disease: str, ratio: float) -> Dict[str, Any]:
    """
    Smart medical report with economic loss estimation.

    Args:
        disease: Predicted disease name
        ratio: Infection ratio percentage

    Returns:
        dict: Structured medical report
    """
    if ratio < 20:
        stage, yield_loss = "early", 5
    elif ratio < 40:
        stage, yield_loss = "moderate", 15
    elif ratio < 60:
        stage, yield_loss = "advanced", 28
    else:
        stage, yield_loss = "critical", 45

    medical = (
        f"The plant shows symptoms of {disease} infection. "
        f"Current severity level is {stage} with an estimated infection ratio of {ratio:.2f}%. "
        f"Photosynthetic activity is being reduced progressively due to tissue necrosis."
    )

    treatment = (
        "Apply systemic fungicide immediately. "
        "Rotate with protectant fungicides every 7-10 days. "
        "Ensure full canopy spray coverage."
    )

    irrigation = (
        "Avoid overhead irrigation. "
        "Prefer drip irrigation to reduce leaf wetness duration. "
        "Maintain balanced soil moisture."
    )

    economic = (
        f"Estimated yield loss may reach about {yield_loss}% if disease progression continues. "
        f"Market quality reduction is expected."
    )

    return {
        "medical": medical,
        "treatment": treatment,
        "irrigation": irrigation,
        "economic_risk": economic,
        "yield_loss_percent": yield_loss,
        "fungicides": [
            {"name": "Mancozeb", "type": "Protectant"},
            {"name": "Difenoconazole", "type": "Systemic"},
        ],
    }


# ============================================================================
# Grad-CAM Research Functions (for Jupyter notebooks / thesis)
# ============================================================================

def apply_gradcam_standalone(
    model: tf.keras.Model,
    img_path: str,
    reverse_class_indices: Dict[int, str],
    layer_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    TensorFlow Grad-CAM for research/notebook use.
    Standalone function for Jupyter notebook analysis.

    Args:
        model: Loaded Keras model (.h5)
        img_path: Path to image file
        reverse_class_indices: {0: "Label", 1: "Label", ...}
        layer_name: Optional specific layer name

    Returns:
        dict with original, heatmap, superimposed PIL Images + metadata
    """
    try:
        import matplotlib.pyplot as plt

        img = Image.open(img_path).convert("RGB")
        original_np = np.array(img)

        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Find last Conv2D layer index
        target_layer_idx = None
        target_layer_name = None
        for i in range(len(model.layers) - 1, -1, -1):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D):
                target_layer_idx = i
                target_layer_name = model.layers[i].name
                break

        if target_layer_idx is None:
            logger.warning("No Conv2D layer found for Grad-CAM")
            return None

        def forward_with_intermediate(x):
            output = x
            conv_output = None
            for idx, layer in enumerate(model.layers):
                output = layer(output, training=False)
                if idx == target_layer_idx:
                    conv_output = output
            return conv_output, output

        with tf.GradientTape() as tape:
            conv_features, predictions = forward_with_intermediate(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_features)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_features_squeezed = conv_features[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_features_squeezed), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        heatmap_np = heatmap.numpy()
        heatmap_np = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_np)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        pred_idx = int(np.argmax(predictions[0]))
        pred_class = reverse_class_indices.get(pred_idx, "Unknown")
        confidence = float(predictions[0][pred_idx]) * 100

        # Display 3-panel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        axes[2].set_title(
            f"Prediction: {pred_class.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%",
            fontsize=14,
            fontweight="bold",
        )
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

        return {
            "original": img,
            "heatmap": Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)),
            "superimposed": Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)),
            "predicted_class": pred_class,
            "confidence": confidence,
            "true_label": os.path.basename(os.path.dirname(img_path)),
            "target_layer": target_layer_name,
        }

    except Exception as e:
        logger.error("Grad-CAM Error: %s: %s", type(e).__name__, e, exc_info=True)
        return None


def generate_gradcam_visualizations(
    model: tf.keras.Model,
    reverse_class_indices: Dict[int, str],
    data_dir: str,
    num_samples: int = 5,
    output_dir: str = "gradcam_outputs",
) -> List[Dict[str, Any]]:
    """
    Generate Grad-CAM for multiple disease classes from dataset.
    For thesis/research documentation.

    Args:
        model: Loaded Keras model
        reverse_class_indices: Class mapping dict
        data_dir: Path to dataset folder (organized by disease class)
        num_samples: Number of random images to visualize
        output_dir: Directory to save visualizations

    Returns:
        list of result dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating Grad-CAM visualizations...")

    sample_images = []
    disease_classes = []

    for disease_folder in os.listdir(data_dir):
        disease_folder_path = os.path.join(data_dir, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        img_files = [
            f
            for f in os.listdir(disease_folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if img_files:
            selected_img = os.path.join(disease_folder_path, random.choice(img_files))
            sample_images.append((selected_img, disease_folder))
            disease_classes.append(disease_folder)

    logger.info("Found %d disease classes with images", len(disease_classes))

    if not sample_images:
        logger.warning("No sample images found.")
        return []

    samples_to_visualize = random.sample(sample_images, min(num_samples, len(sample_images)))
    results = []

    for i, (img_path, true_label) in enumerate(samples_to_visualize, 1):
        logger.info("Sample %d/%d - %s", i, len(samples_to_visualize), true_label)

        result = apply_gradcam_standalone(model, img_path, reverse_class_indices)

        if result:
            results.append(result)

            safe_label = true_label.replace("/", "_").replace("__", "_").replace(" ", "_")
            save_filename = f"gradcam_sample_{i:02d}_{safe_label}.png"
            save_path = os.path.join(output_dir, save_filename)

            result["superimposed"].save(save_path, dpi=(300, 300), quality=95)

            match = (
                true_label.lower() in result["predicted_class"].lower()
                or result["predicted_class"].lower() in true_label.lower()
            )
            status = "CORRECT" if match else "MISMATCH"
            logger.info(
                "  Prediction: %s (%.1f%%) %s",
                result["predicted_class"],
                result["confidence"],
                status,
            )

    correct = sum(
        1
        for r in results
        if r["true_label"].lower() in r["predicted_class"].lower()
        or r["predicted_class"].lower() in r["true_label"].lower()
    )
    accuracy = (correct / len(results)) * 100 if results else 0
    logger.info("Completed %d visualizations. Accuracy: %d/%d (%.1f%%)", len(results), correct, len(results), accuracy)

    return results


def export_gradcam_report(
    results: List[Dict[str, Any]],
    model: tf.keras.Model,
    reverse_class_indices: Dict[int, str],
    output_file: str = "gradcam_thesis_report.json",
) -> Dict[str, Any]:
    """
    Export Grad-CAM results to JSON for thesis documentation.

    Args:
        results: List of result dicts from generate_gradcam_visualizations()
        model: Keras model
        reverse_class_indices: Class mapping
        output_file: Output JSON file path

    Returns:
        dict: Complete report
    """
    correct_count = sum(
        1
        for r in results
        if r["true_label"].lower() in r["predicted_class"].lower()
    )

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model_input_shape": str(model.input_shape),
            "num_classes": len(reverse_class_indices),
            "tensorflow_version": tf.__version__,
        },
        "gradcam_parameters": {
            "target_layer": "Last Conv2D layer",
            "colormap": "JET (blue->red = low->high activation)",
            "blend_ratio": "60% original + 40% heatmap",
            "output_resolution": "300 DPI",
        },
        "visualizations": [
            {
                "sample_number": i + 1,
                "true_label": r["true_label"],
                "predicted_class": r["predicted_class"],
                "confidence_percent": round(r["confidence"], 2),
                "match": r["true_label"].lower() in r["predicted_class"].lower(),
            }
            for i, r in enumerate(results)
        ],
        "accuracy": {
            "correct_predictions": correct_count,
            "total_samples": len(results),
            "accuracy_percent": round(correct_count / len(results) * 100 if results else 0, 2),
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report exported: %s", output_file)
    return report


# ============================================================================
# End of File
# ============================================================================