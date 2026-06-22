# ============================================================================
# analysis/api_views.py
# AJAX API Endpoints — Prediction, Grad-CAM, AI Reports
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .validators import validate_image_upload, sanitize_filename
from .services import (
    get_classifier,
    generate_gradcam,
    estimate_disease_progress,
    draw_red_regions_boxes,
    ai_treatment_advisor,
    ai_doctor_report,
    prediction_logger,
    PredictionLogEntry,
    performance_tracker,
    log_error,
)

logger = logging.getLogger(__name__)


# ============================================================================
# AJAX Image Prediction
# ============================================================================

@require_POST
def api_predict(request):
    """
    POST /api/predict/
    AJAX endpoint: upload image → prediction + Grad-CAM.

    Returns JSON with prediction results and image URLs.
    Fully offline — no external API calls.
    """
    if not request.FILES.get("image"):
        return JsonResponse({"error": "No image file provided."}, status=400)

    image_file = request.FILES["image"]

    # Server-side validation
    try:
        validate_image_upload(image_file)
    except ValidationError as e:
        return JsonResponse({"error": str(e.message)}, status=400)

    try:
        classifier = get_classifier()
        fs = FileSystemStorage()
        uid = uuid.uuid4().hex[:8]
        clean_name = sanitize_filename(image_file.name)

        # --- Prediction ---
        image_file.seek(0)
        result = classifier.predict(image_file)
        result_dict = result.to_dict()

        # --- Save original image ---
        image_file.seek(0)
        original_name = f"{uid}_orig_{clean_name}"
        original_path = fs.save(original_name, image_file)
        original_url = fs.url(original_path)

        # --- Grad-CAM ---
        image_file.seek(0)
        gradcam = generate_gradcam(classifier.model, image_file)

        gradcam_url = None
        progress_ratio = 0.0
        progress_stage = "Unknown"
        activation_stats = None

        if gradcam:
            # Draw bounding boxes on superimposed image
            boxed = draw_red_regions_boxes(gradcam["superimposed"])
            display_img = boxed if boxed else gradcam["superimposed"]

            gradcam_name = f"{uid}_gradcam.png"
            gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_name)
            display_img.save(gradcam_path, format="PNG", optimize=True)
            gradcam_url = settings.MEDIA_URL + gradcam_name

            # Disease progress
            progress_ratio, progress_stage = estimate_disease_progress(
                gradcam["superimposed"]
            )

            activation_stats = gradcam.get("activation_stats")

            # AI treatment advisor (local, offline)
            recommendations = ai_treatment_advisor(
                result.disease, result.confidence, progress_stage
            )
            result_dict["recommendations"] = recommendations

        # --- Log prediction ---
        try:
            user_id = request.user.id if request.user.is_authenticated else None
            log_entry = PredictionLogEntry(
                timestamp=datetime.utcnow().isoformat() + "Z",
                disease=result.disease,
                confidence=result.confidence,
                is_healthy=result.is_healthy,
                is_low_confidence=result.is_low_confidence,
                inference_time_ms=result.inference_time_ms,
                gradcam_time_ms=gradcam.get("gradcam_time_ms", 0) if gradcam else 0,
                image_size_bytes=image_file.size,
                stage=progress_stage,
                disease_ratio=progress_ratio,
                user_id=user_id,
            )
            prediction_logger.log(log_entry)
        except Exception as e:
            logger.warning("Failed to log prediction: %s", e)

        return JsonResponse({
            "status": "ok",
            "result": result_dict,
            "image_url": original_url,
            "gradcam_url": gradcam_url,
            "progress_ratio": progress_ratio,
            "progress_stage": progress_stage,
            "activation_stats": activation_stats,
        })

    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        log_error(e, {"endpoint": "api_predict"})
        return JsonResponse(
            {"error": "Analysis failed. Please try a different image."},
            status=500,
        )


# ============================================================================
# Lazy-loaded AI Report (optional — uses external API if available)
# ============================================================================

@require_POST
def api_ai_report(request):
    """
    POST /api/predict/ai-report/
    Generate an AI doctor report for a prediction.

    Primary: local rule-based report (always works offline).
    Optional: enhanced Gemini report if API is available.
    """
    try:
        data = json.loads(request.body)
        disease = data.get("disease", "").strip()
        confidence = float(data.get("confidence", 0))
        stage = data.get("stage", "Unknown")
        ratio = float(data.get("ratio", 0))

        if not disease:
            return JsonResponse({"error": "Disease name is required."}, status=400)

        # Local rule-based report (always available, fully offline)
        report = ai_doctor_report(disease, ratio)

        # Optional: try Gemini for enhanced report
        gemini_report = None
        try:
            from .services.ai_service import get_gemini_service
            service = get_gemini_service()
            if service.is_available:
                gemini_result = service.generate_treatment_plan(
                    disease, stage, str(ratio)
                )
                if "text" in gemini_result:
                    gemini_report = gemini_result["text"]
        except Exception as e:
            logger.warning("Gemini report unavailable: %s", e)

        return JsonResponse({
            "status": "ok",
            "report": report,
            "gemini_report": gemini_report,
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    except Exception as e:
        logger.error("AI report error: %s", e, exc_info=True)
        return JsonResponse({"error": "AI report generation failed."}, status=500)


# ============================================================================
# Performance Stats
# ============================================================================

@login_required
def api_stats(request):
    """
    GET /api/stats/
    Returns system performance statistics.
    """
    stats = performance_tracker.get_stats()
    recent = prediction_logger.get_recent(20)

    return JsonResponse({
        "status": "ok",
        "performance": stats,
        "recent_predictions": recent,
    })
