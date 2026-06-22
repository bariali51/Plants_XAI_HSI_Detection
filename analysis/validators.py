# ============================================================================
# analysis/validators.py
# Input Validation Utilities
# ============================================================================

from __future__ import annotations

import os
import re
from typing import Optional

from django.core.exceptions import ValidationError


# ============================================================================
# Image Upload Validation
# ============================================================================

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def validate_image_upload(uploaded_file) -> None:
    """
    Validate an uploaded image file from Django request.FILES.

    Checks:
        1. File exists and is not empty
        2. File size within limit
        3. Content type is an allowed image type
        4. File extension matches an allowed type

    Args:
        uploaded_file: Django UploadedFile

    Raises:
        ValidationError: If any check fails
    """
    if not uploaded_file:
        raise ValidationError("No file was uploaded.")

    # Size check
    if uploaded_file.size == 0:
        raise ValidationError("Uploaded file is empty.")

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        mb = MAX_UPLOAD_SIZE / (1024 * 1024)
        raise ValidationError(f"File exceeds maximum size of {mb:.0f} MB.")

    # Content type check
    content_type = getattr(uploaded_file, "content_type", "")
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(
            f"Invalid file type '{content_type}'. "
            f"Allowed types: JPEG, PNG, WebP, BMP, TIFF."
        )

    # Extension check
    _, ext = os.path.splitext(uploaded_file.name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Invalid file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )


# ============================================================================
# Filename Sanitization
# ============================================================================

def sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    Sanitize a filename for safe storage.

    - Strips directory components (path traversal prevention)
    - Removes dangerous characters
    - Limits length
    """
    # Strip directory components
    name = os.path.basename(name)
    # Remove anything that's not alphanumeric, dash, underscore, dot, or space
    name = re.sub(r"[^\w\-_. ]", "_", name)
    # Collapse multiple underscores/spaces
    name = re.sub(r"[_\s]+", "_", name)
    # Trim leading/trailing underscores
    name = name.strip("_")
    # Limit length preserving extension
    if len(name) > max_length:
        base, ext = os.path.splitext(name)
        name = base[: max_length - len(ext)] + ext
    return name or "upload"


# ============================================================================
# Scan Input Validation
# ============================================================================

def validate_scan_save_data(data: dict) -> dict:
    """
    Validate and sanitize POST data for saving a scan.

    Args:
        data: QueryDict or dict from request.POST

    Returns:
        dict of cleaned data

    Raises:
        ValidationError: If required fields are missing or invalid
    """
    cleaned = {}

    # Required fields
    prediction = data.get("prediction", "").strip()
    if not prediction:
        raise ValidationError("Prediction is required.")
    cleaned["prediction"] = prediction[:120]  # max length

    # Confidence
    try:
        confidence_str = data.get("confidence", "0").replace("%", "")
        cleaned["confidence"] = min(max(float(confidence_str), 0), 100)
    except (ValueError, TypeError):
        cleaned["confidence"] = 0.0

    # Ratio
    try:
        cleaned["disease_ratio"] = min(max(float(data.get("ratio", "0")), 0), 100)
    except (ValueError, TypeError):
        cleaned["disease_ratio"] = 0.0

    # Stage
    stage = data.get("stage", "Unknown").strip()
    allowed_stages = {"Very Early", "Early", "Moderate", "Advanced", "Severe", "Unknown"}
    cleaned["disease_stage"] = stage if stage in allowed_stages else "Unknown"

    # Folder name
    folder_name = data.get("folder_name", "Untitled").strip()
    cleaned["folder_name"] = sanitize_filename(folder_name, max_length=120)

    # Optional fields
    cleaned["ai_medical"] = (data.get("ai_medical", "") or "")[:2000]
    cleaned["ai_treatment"] = (data.get("ai_treatment", "") or "")[:2000]
    cleaned["ai_irrigation"] = (data.get("ai_irrigation", "") or "")[:2000]
    cleaned["ai_economic"] = (data.get("ai_economic", "") or "")[:2000]

    try:
        cleaned["yield_loss"] = min(max(float(data.get("yield_loss", "0")), 0), 100)
    except (ValueError, TypeError):
        cleaned["yield_loss"] = 0.0

    cleaned["fungicides_json"] = data.get("fungicides_json", "[]")
    cleaned["orig"] = data.get("orig", "")
    cleaned["gradcam"] = data.get("gradcam", "")

    return cleaned
