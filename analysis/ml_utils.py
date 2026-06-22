# ============================================================================
# analysis/ml_utils.py
# Plant Disease Detection with XAI (Grad-CAM)
# PyTorch Implementation — Optimized
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
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

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
MODEL_PATH = os.path.join(BASE_DIR, "model_files", "efficientnet_b0_plant_disease.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model_files", "class_names.json")
TREATMENTS_PATH = os.path.join(BASE_DIR, "model_files", "treatments.json")

# Model expects 224x224 input
IMG_SIZE: Tuple[int, int] = (224, 224)

# Standard ImageNet normalization
_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    """PyTorch-based plant disease classifier with Grad-CAM explainability."""

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.reverse_class_indices: Optional[Dict[int, str]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_resources()

    def load_resources(self) -> None:
        """Load PyTorch model and class names."""
        logger.info("Loading PyTorch model...")

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        self.reverse_class_indices = {int(k): v for k, v in class_names.items()}
        num_classes = len(self.reverse_class_indices)

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded: %d classes, device=%s", num_classes, self.device)

    def preprocess_image(self, image_file) -> Tuple[torch.Tensor, Image.Image]:
        """Prepare image for PyTorch model inference."""
        image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        img_tensor = _transform(img_resized).unsqueeze(0).to(self.device)
        return img_tensor, img_resized

    def predict(self, image_file) -> Dict[str, Any]:
        """Predict disease from uploaded image."""
        img_tensor, _ = self.preprocess_image(image_file)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predictions = probabilities[0].cpu().numpy()

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
        if lookup_key in TREATMENT_RECOMMENDATIONS:
            return TREATMENT_RECOMMENDATIONS[lookup_key]
        for key in TREATMENT_RECOMMENDATIONS:
            if key in lookup_key or lookup_key in key:
                return TREATMENT_RECOMMENDATIONS[key]
        plant = raw_class.split("__")[0].lower() if "__" in raw_class else raw_class.lower()
        for key in TREATMENT_RECOMMENDATIONS:
            if plant in key:
                return TREATMENT_RECOMMENDATIONS[key]
        return ["Consult an agricultural specialist for specific treatment."]

    def _find_last_conv_layer(self) -> Optional[Tuple[str, nn.Module]]:
        """Find the last Conv2d layer name and module."""
        last_name, last_module = None, None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name
                last_module = module
        return (last_name, last_module) if last_name else None

    def apply_gradcam(self, image_file, layer_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate Grad-CAM heatmap for PyTorch model."""
        try:
            image_file.seek(0)
            original_img = Image.open(image_file).convert("RGB")
            original_np = np.array(original_img)

            image_file.seek(0)
            img_tensor, _ = self.preprocess_image(image_file)
            img_tensor.requires_grad_(True)

            result = self._find_last_conv_layer()
            if result is None:
                logger.warning("No Conv2d layer found for Grad-CAM")
                return None
            target_layer_name, target_module = result

            activations, gradients_store = {}, {}

            def fwd_hook(m, i, o):
                activations["value"] = o

            def bwd_hook(m, gi, go):
                gradients_store["value"] = go[0]

            fh = target_module.register_forward_hook(fwd_hook)
            bh = target_module.register_full_backward_hook(bwd_hook)

            outputs = self.model(img_tensor)
            class_idx = outputs.argmax(dim=1).item()
            self.model.zero_grad()
            outputs[0, class_idx].backward()

            fh.remove()
            bh.remove()

            grads = gradients_store["value"]
            acts = activations["value"]
            pooled = torch.mean(grads, dim=(0, 2, 3))

            for i in range(acts.shape[1]):
                acts[0, i, :, :] *= pooled[i]

            heatmap = torch.mean(acts[0], dim=0)
            heatmap = torch.clamp(heatmap, min=0)
            heatmap = heatmap / (heatmap.max() + 1e-8)
            heatmap_np = heatmap.detach().cpu().numpy()

            heatmap_np = cv2.resize(heatmap_np, (original_np.shape[1], original_np.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_np)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
            superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

            pred_class = self.reverse_class_indices.get(class_idx, "Unknown")

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
    """Estimate disease spread ratio based on Grad-CAM heatmap colors."""
    if gradcam_pil is None:
        return 0.0, "Unknown"
    img = np.array(gradcam_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        + cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    )
    yellow_mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([35, 255, 255]))
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
    """Draw bounding boxes around red (high-activation) regions in the heatmap."""
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


def ai_treatment_advisor(disease_name: str, confidence: float, stage: str) -> List[str]:
    """Smart treatment advisor based on disease type and severity stage."""
    disease_lower = disease_name.lower()
    advice: List[str] = []
    if "early_blight" in disease_lower:
        advice.extend(["Remove infected leaves immediately to prevent spread.", "Apply copper-based fungicide every 7 days.", "Avoid overhead irrigation to reduce humidity."])
    elif "late_blight" in disease_lower:
        advice.extend(["Isolate affected plants urgently.", "Use systemic fungicides containing metalaxyl.", "Improve field drainage and airflow."])
    elif "bacterial_spot" in disease_lower:
        advice.extend(["Use certified disease-free seeds.", "Spray copper bactericides weekly.", "Rotate crops next season."])
    elif "leaf_mold" in disease_lower:
        advice.extend(["Increase greenhouse ventilation.", "Reduce leaf wetness duration.", "Apply preventive fungicide."])
    elif "healthy" in disease_lower:
        advice.extend(["Plant appears healthy.", "Maintain balanced fertilization.", "Monitor regularly for early symptoms."])
    else:
        advice.extend(["Consult agricultural specialist.", "Monitor disease progression closely.", "Apply broad-spectrum fungicide if necessary."])
    if stage == "Advanced":
        advice.append("Disease is advanced - immediate chemical control recommended.")
    if confidence > 90:
        advice.append("AI confidence is high - treatment should start immediately.")
    return advice


def ai_llm_doctor(disease: str, confidence: float, stage: str, ratio: float) -> str:
    """Medical consultation via OpenAI GPT, with rich local fallback."""
    try:
        from openai import OpenAI
        from django.conf import settings
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = f"\nYou are an expert plant disease doctor.\n\nDisease: {disease}\nConfidence: {confidence}%\nStage: {stage}\nSeverity ratio: {ratio}%\n\nGive professional treatment explanation and agronomic advice.\n"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a plant pathology expert."}, {"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=250,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.debug("OpenAI unavailable, using local model: %s", e)
        return _build_local_doctor_response(disease, confidence, stage, ratio)


def _build_local_doctor_response(disease: str, confidence: float, stage: str, ratio: float) -> str:
    """Build a rich AI Doctor response from the local knowledge base."""
    disease_lower = disease.lower().strip().replace(" ", "_")

    # ── Determine severity level and yield impact ──
    if ratio >= 60:
        severity = "critical"
        yield_loss = 45
        urgency = "⚠️ URGENT — Immediate intervention required!"
    elif ratio >= 40:
        severity = "advanced"
        yield_loss = 28
        urgency = "⚡ Action needed within 24–48 hours."
    elif ratio >= 20:
        severity = "moderate"
        yield_loss = 15
        urgency = "📋 Begin treatment within the next few days."
    else:
        severity = "early"
        yield_loss = 5
        urgency = "👀 Monitor closely and apply preventive measures."

    sections = [
        f"🩺 **Nabtati AI Doctor — Offline Analysis**\n",
        f"**Diagnosis:** {disease.replace('_', ' ').title()}",
        f"**Confidence:** {confidence:.1f}%",
        f"**Stage:** {stage} ({severity.title()})",
        f"**Infection ratio:** {ratio:.1f}%",
        f"\n{urgency}\n",
    ]

    # ── Treatment recommendations from knowledge base ──
    treatments = TREATMENT_RECOMMENDATIONS.get(disease_lower)
    if not treatments:
        # Partial key matching
        for key in TREATMENT_RECOMMENDATIONS:
            if key in disease_lower or disease_lower in key:
                treatments = TREATMENT_RECOMMENDATIONS[key]
                break
    if not treatments:
        # Try matching by plant name (first part of label)
        plant = disease_lower.split("__")[0] if "__" in disease_lower else disease_lower.split("_")[0]
        for key in TREATMENT_RECOMMENDATIONS:
            if plant in key:
                treatments = TREATMENT_RECOMMENDATIONS[key]
                break

    if treatments:
        sections.append("---\n**💊 Treatment Plan:**\n")
        for i, t in enumerate(treatments, 1):
            sections.append(f"{i}. {t}")
    else:
        sections.append("---\n**💊 General Treatment Guidance:**\n")
        sections.append("1. Apply broad-spectrum fungicide as preventive measure.")
        sections.append("2. Remove and destroy visibly infected plant tissue.")
        sections.append("3. Improve airflow and spacing between plants.")

    # ── Irrigation advice ──
    sections.append("\n---\n**💧 Irrigation Management:**\n")
    sections.append("• Avoid overhead watering — prefer drip irrigation.")
    sections.append("• Water early morning to reduce leaf wetness duration.")
    sections.append("• Maintain balanced soil moisture without waterlogging.")

    # ── Severity-specific actions ──
    sections.append(f"\n---\n**📋 Severity-Based Actions ({severity.title()}):**\n")
    if severity == "critical":
        sections.extend([
            "• Isolate affected plants immediately to stop spread.",
            "• Remove and destroy severely infected tissue.",
            "• Apply systemic fungicide (e.g., Difenoconazole) within 24 hours.",
            "• Consult an agricultural specialist urgently.",
            "• Monitor surrounding plants daily for new symptoms.",
        ])
    elif severity == "advanced":
        sections.extend([
            "• Begin targeted fungicide application (alternate systemic + protectant).",
            "• Improve air circulation around plants.",
            "• Adjust irrigation to reduce leaf wetness.",
            "• Schedule follow-up scan in 3 days to track progress.",
        ])
    elif severity == "moderate":
        sections.extend([
            "• Apply preventive fungicide (e.g., Mancozeb).",
            "• Monitor plant daily for signs of progression.",
            "• Ensure proper nutrient supply (balanced NPK).",
            "• Re-scan in 5–7 days to compare evolution.",
        ])
    else:
        sections.extend([
            "• Continue regular monitoring schedule.",
            "• Apply preventive bio-fungicide as a precaution.",
            "• Maintain optimal growing conditions (temperature, humidity).",
            "• Document with follow-up scans for comparison.",
        ])

    # ── Economic impact ──
    sections.append(f"\n---\n**📊 Economic Impact Estimate:**\n")
    sections.append(f"• Estimated yield loss: ~{yield_loss}% if untreated.")
    sections.append(f"• Market quality reduction is {'likely' if ratio > 30 else 'possible'}.")
    sections.append(f"• Recommended fungicides: Mancozeb (protectant), Difenoconazole (systemic).")

    # ── Footer ──
    sections.append(
        "\n---\n_🌿 This report was generated by the Nabtati local AI model. "
        "Online AI doctor will reconnect automatically when available._"
    )

    return "\n".join(sections)


def ai_gemini_doctor(disease: str, confidence: float, stage: str, ratio: float) -> Dict[str, Any]:
    """Medical consultation via Google Gemini."""
    try:
        import google.generativeai as genai
        from django.conf import settings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-flash-latest")
        prompt = f"\nYou are a professional plant disease agronomist AI.\n\nGive response STRICTLY in this format:\n\nMEDICAL:\n(short scientific diagnosis)\n\nTREATMENT:\n(clear fungicide and agronomic actions)\n\nIRRIGATION:\n(water management advice)\n\nECONOMIC:\n(yield loss risk and financial impact)\n\nDisease: {disease}\nConfidence: {confidence}\nStage: {stage}\nInfection ratio: {ratio}%\n"
        response = model.generate_content(prompt)
        text = response.text
        sections = text.split("TREATMENT:")
        medical_text = sections[0].replace("MEDICAL:", "").strip() if sections else "N/A"
        rest = sections[1].split("IRRIGATION:") if len(sections) > 1 else ["", ""]
        treatment_text = rest[0].strip()
        rest2 = rest[1].split("ECONOMIC:") if len(rest) > 1 else ["", ""]
        irrigation_text = rest2[0].strip()
        risk_text = rest2[1].strip() if len(rest2) > 1 else "N/A"
        yield_loss = int(ratio * 0.6)
        return {"medical": medical_text, "treatment": treatment_text, "irrigation": irrigation_text, "economic_risk": risk_text, "yield_loss_percent": yield_loss, "fungicides": [{"name": "Mancozeb", "type": "Protectant"}, {"name": "Difenoconazole", "type": "Systemic"}]}
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return {"medical": "AI doctor unavailable", "treatment": "-", "irrigation": "-", "economic_risk": "-", "yield_loss_percent": 0, "fungicides": []}


def ai_compare_evolution(old_disease: str, old_ratio: float, new_ratio: float) -> str:
    """Compare disease evolution between two scans."""
    diff = new_ratio - old_ratio
    if diff > 15:
        trend, status = "Disease has progressed aggressively.", "Severe deterioration"
    elif diff > 5:
        trend, status = "Disease progression detected.", "Condition worsening"
    elif diff > -5:
        trend, status = "Disease remains relatively stable.", "Stable"
    else:
        trend, status = "Disease regression observed. Plant health improving.", "Improvement"
    return f"\nAI Evolution Analysis:\n\nPlant disease: {old_disease}\n\nPrevious infection level: {old_ratio:.2f}%\nCurrent infection level: {new_ratio:.2f}%\n\nOverall evolution status: {status}\n\nInterpretation:\n{trend}\n\nRecommendation:\nContinuous monitoring is strongly advised.\nAdjust fungicide program according to disease dynamics.\nProtect remaining healthy foliage to secure yield potential.\n"


def ai_doctor_report(disease: str, ratio: float) -> Dict[str, Any]:
    """Smart medical report with economic loss estimation."""
    if ratio < 20:
        stage, yield_loss = "early", 5
    elif ratio < 40:
        stage, yield_loss = "moderate", 15
    elif ratio < 60:
        stage, yield_loss = "advanced", 28
    else:
        stage, yield_loss = "critical", 45
    return {
        "medical": f"The plant shows symptoms of {disease} infection. Current severity level is {stage} with an estimated infection ratio of {ratio:.2f}%. Photosynthetic activity is being reduced progressively due to tissue necrosis.",
        "treatment": "Apply systemic fungicide immediately. Rotate with protectant fungicides every 7-10 days. Ensure full canopy spray coverage.",
        "irrigation": "Avoid overhead irrigation. Prefer drip irrigation to reduce leaf wetness duration. Maintain balanced soil moisture.",
        "economic_risk": f"Estimated yield loss may reach about {yield_loss}% if disease progression continues. Market quality reduction is expected.",
        "yield_loss_percent": yield_loss,
        "fungicides": [{"name": "Mancozeb", "type": "Protectant"}, {"name": "Difenoconazole", "type": "Systemic"}],
    }


# ============================================================================
# Grad-CAM Research Functions (for Jupyter notebooks / thesis)
# ============================================================================

def apply_gradcam_standalone(
    model: nn.Module,
    img_path: str,
    reverse_class_indices: Dict[int, str],
    layer_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """PyTorch Grad-CAM for research/notebook use."""
    try:
        import matplotlib.pyplot as plt

        img = Image.open(img_path).convert("RGB")
        original_np = np.array(img)
        img_resized = img.resize(IMG_SIZE)
        img_tensor = _transform(img_resized).unsqueeze(0)

        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)
        img_tensor.requires_grad_(True)

        # Find last conv layer
        target_name, target_module = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_name = name
                target_module = module
        if target_module is None:
            logger.warning("No Conv2d layer found for Grad-CAM")
            return None

        activations, grads_store = {}, {}
        fh = target_module.register_forward_hook(lambda m, i, o: activations.update({"v": o}))
        bh = target_module.register_full_backward_hook(lambda m, gi, go: grads_store.update({"v": go[0]}))

        outputs = model(img_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        model.zero_grad()
        outputs[0, pred_idx].backward()
        fh.remove()
        bh.remove()

        pooled = torch.mean(grads_store["v"], dim=(0, 2, 3))
        acts = activations["v"]
        for i in range(acts.shape[1]):
            acts[0, i] *= pooled[i]
        heatmap = torch.clamp(torch.mean(acts[0], dim=0), min=0)
        heatmap = (heatmap / (heatmap.max() + 1e-8)).detach().cpu().numpy()
        heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = float(probs[0][pred_idx]) * 100
        pred_class = reverse_class_indices.get(pred_idx, "Unknown")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight="bold")
        axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Prediction: {pred_class.replace('_', ' ').title()}\nConfidence: {confidence:.1f}%", fontsize=14, fontweight="bold")
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()

        return {
            "original": img, "heatmap": Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)),
            "superimposed": Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)),
            "predicted_class": pred_class, "confidence": confidence,
            "true_label": os.path.basename(os.path.dirname(img_path)), "target_layer": target_name,
        }
    except Exception as e:
        logger.error("Grad-CAM Error: %s: %s", type(e).__name__, e, exc_info=True)
        return None


def generate_gradcam_visualizations(
    model: nn.Module, reverse_class_indices: Dict[int, str],
    data_dir: str, num_samples: int = 5, output_dir: str = "gradcam_outputs",
) -> List[Dict[str, Any]]:
    """Generate Grad-CAM for multiple disease classes from dataset."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating Grad-CAM visualizations...")
    sample_images, disease_classes = [], []

    for disease_folder in os.listdir(data_dir):
        disease_folder_path = os.path.join(data_dir, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue
        img_files = [f for f in os.listdir(disease_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
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
            save_path = os.path.join(output_dir, f"gradcam_sample_{i:02d}_{safe_label}.png")
            result["superimposed"].save(save_path, dpi=(300, 300), quality=95)
            match = true_label.lower() in result["predicted_class"].lower() or result["predicted_class"].lower() in true_label.lower()
            logger.info("  Prediction: %s (%.1f%%) %s", result["predicted_class"], result["confidence"], "CORRECT" if match else "MISMATCH")

    correct = sum(1 for r in results if r["true_label"].lower() in r["predicted_class"].lower() or r["predicted_class"].lower() in r["true_label"].lower())
    accuracy = (correct / len(results)) * 100 if results else 0
    logger.info("Completed %d visualizations. Accuracy: %d/%d (%.1f%%)", len(results), correct, len(results), accuracy)
    return results


def export_gradcam_report(
    results: List[Dict[str, Any]], model: nn.Module,
    reverse_class_indices: Dict[int, str], output_file: str = "gradcam_thesis_report.json",
) -> Dict[str, Any]:
    """Export Grad-CAM results to JSON for thesis documentation."""
    correct_count = sum(1 for r in results if r["true_label"].lower() in r["predicted_class"].lower())
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_classes": len(reverse_class_indices),
            "pytorch_version": torch.__version__,
        },
        "gradcam_parameters": {"target_layer": "Last Conv2d layer", "colormap": "JET (blue->red = low->high activation)", "blend_ratio": "60% original + 40% heatmap", "output_resolution": "300 DPI"},
        "visualizations": [{"sample_number": i + 1, "true_label": r["true_label"], "predicted_class": r["predicted_class"], "confidence_percent": round(r["confidence"], 2), "match": r["true_label"].lower() in r["predicted_class"].lower()} for i, r in enumerate(results)],
        "accuracy": {"correct_predictions": correct_count, "total_samples": len(results), "accuracy_percent": round(correct_count / len(results) * 100 if results else 0, 2)},
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report exported: %s", output_file)
    return report


# ============================================================================
# End of File
# ============================================================================