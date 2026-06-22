# ============================================================================
# analysis/services/treatment_service.py
# Treatment Recommendations & AI Doctor Reports (Fully Offline)
# ============================================================================

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Treatment Data — Loaded once at module import
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TREATMENTS_PATH = os.path.join(_BASE_DIR, "model_files", "treatments.json")
_CLASS_INDICES_PATH = os.path.join(_BASE_DIR, "model_files", "class_indices.json")

with open(_TREATMENTS_PATH, "r", encoding="utf-8") as _f:
    _treatment_data = json.load(_f)

TREATMENT_RECOMMENDATIONS: Dict[str, List[str]] = _treatment_data["TREATMENT_RECOMMENDATIONS"]
DEFAULT_HEALTHY_PRACTICES: List[str] = _treatment_data["DEFAULT_HEALTHY_PRACTICES"]

# ---------------------------------------------------------------------------
# Build normalized lookup index for O(1) matching
# ---------------------------------------------------------------------------
# Normalize all keys: lowercase, strip, collapse spaces/underscores
_NORMALIZED_INDEX: Dict[str, List[str]] = {}
for _key, _treatments in TREATMENT_RECOMMENDATIONS.items():
    _norm = _key.lower().strip().replace(" ", "_")
    _NORMALIZED_INDEX[_norm] = _treatments

# Validate coverage at startup (log warnings for uncovered classes)
try:
    with open(_CLASS_INDICES_PATH, "r", encoding="utf-8") as _f2:
        _class_indices = json.load(_f2)
    _uncovered = []
    for _idx, _label in _class_indices.items():
        _norm_label = _label.lower().strip().replace(" ", "_")
        if _norm_label not in _NORMALIZED_INDEX:
            _uncovered.append(_label)
    if _uncovered:
        logger.warning(
            "Treatment coverage gap: %d/%d classes lack dedicated treatments: %s",
            len(_uncovered),
            len(_class_indices),
            ", ".join(_uncovered[:10]) + ("..." if len(_uncovered) > 10 else ""),
        )
    else:
        logger.info(
            "Treatment coverage: all %d classes have dedicated treatments",
            len(_class_indices),
        )
except Exception as _e:
    logger.warning("Could not validate treatment coverage: %s", _e)


# ============================================================================
# Treatment Lookup
# ============================================================================

def get_treatment(predicted_label: str) -> List[str]:
    """Get treatment recommendations for a predicted disease label."""
    norm = predicted_label.lower().strip().replace(" ", "_")
    return _NORMALIZED_INDEX.get(norm, DEFAULT_HEALTHY_PRACTICES)


def find_recommendations(lookup_key: str, raw_class: str) -> List[str]:
    """
    Flexible recommendation lookup with fallback chain.

    Tries: exact normalized match → partial match → plant-type match → fallback.

    Args:
        lookup_key: Normalized lowercase key (e.g., 'tomato__early_blight')
        raw_class: Original class label from class_indices.json

    Returns:
        List of treatment recommendation strings
    """
    # 1. Direct normalized match (O(1))
    if lookup_key in _NORMALIZED_INDEX:
        return _NORMALIZED_INDEX[lookup_key]

    # 2. Partial match — check if any key contains or is contained by lookup_key
    for key, treatments in _NORMALIZED_INDEX.items():
        if key in lookup_key or lookup_key in key:
            return treatments

    # 3. Plant type match — extract plant name before '__'
    plant = raw_class.split("__")[0].lower() if "__" in raw_class else raw_class.lower()
    for key in _NORMALIZED_INDEX:
        if plant in key:
            return _NORMALIZED_INDEX[key]

    # 4. Final fallback
    return ["Consult an agricultural specialist for specific treatment."]


# ============================================================================
# AI Treatment Advisor (Rule-based — Fully Offline)
# ============================================================================

def ai_treatment_advisor(
    disease_name: str, confidence: float, stage: str
) -> List[str]:
    """
    Smart treatment advisor based on disease type and severity stage.

    This is a rule-based system that works fully offline.
    Provides context-specific advice based on disease keywords and severity.

    Args:
        disease_name: Predicted disease name
        confidence: Prediction confidence (0-100)
        stage: Disease severity stage

    Returns:
        list of treatment advice strings
    """
    disease_lower = disease_name.lower()
    advice: List[str] = []

    # Disease-specific advice based on keyword matching
    if "early_blight" in disease_lower or "early blight" in disease_lower:
        advice.extend([
            "Remove infected leaves immediately to prevent spread.",
            "Apply copper-based fungicide every 7 days.",
            "Avoid overhead irrigation to reduce humidity.",
        ])
    elif "late_blight" in disease_lower or "late blight" in disease_lower:
        advice.extend([
            "Isolate affected plants urgently.",
            "Use systemic fungicides containing metalaxyl.",
            "Improve field drainage and airflow.",
        ])
    elif "bacterial_spot" in disease_lower or "bacterial spot" in disease_lower:
        advice.extend([
            "Use certified disease-free seeds.",
            "Spray copper bactericides weekly.",
            "Rotate crops next season.",
        ])
    elif "leaf_mold" in disease_lower or "leaf mold" in disease_lower:
        advice.extend([
            "Increase greenhouse ventilation.",
            "Reduce leaf wetness duration.",
            "Apply preventive fungicide.",
        ])
    elif "rust" in disease_lower:
        advice.extend([
            "Apply triazole-based fungicide promptly.",
            "Remove heavily infected leaves.",
            "Ensure adequate plant spacing for airflow.",
        ])
    elif "powdery_mildew" in disease_lower or "powdery mildew" in disease_lower:
        advice.extend([
            "Apply sulfur-based or potassium bicarbonate spray.",
            "Improve air circulation around plants.",
            "Avoid overhead watering.",
        ])
    elif "mosaic" in disease_lower or "virus" in disease_lower:
        advice.extend([
            "Remove and destroy infected plants immediately.",
            "Control insect vectors (aphids, whiteflies).",
            "Use virus-resistant cultivars for replanting.",
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

    # Severity-based additions
    if stage in ("Advanced", "Severe"):
        advice.append("Disease is advanced — immediate chemical control recommended.")
    elif stage == "Moderate":
        advice.append("Disease is moderate — begin treatment within 48 hours.")

    if confidence > 90:
        advice.append("AI confidence is high — treatment should start immediately.")
    elif confidence < 40:
        advice.append(
            "⚠ AI confidence is low — consider re-scanning with a clearer image "
            "or consulting an expert."
        )

    return advice


# ============================================================================
# AI Doctor Report (Local — Fully Offline, No External API)
# ============================================================================

def ai_doctor_report(disease: str, ratio: float) -> Dict[str, Any]:
    """
    Generate a structured medical report with economic loss estimation.

    Fully offline — uses rule-based logic, no external API calls.

    Args:
        disease: Predicted disease name
        ratio: Infection ratio percentage

    Returns:
        dict with medical, treatment, irrigation, economic_risk,
              yield_loss_percent, and fungicides
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
# Evolution Comparison (Fully Offline)
# ============================================================================

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

    return f"""AI Evolution Analysis:

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
