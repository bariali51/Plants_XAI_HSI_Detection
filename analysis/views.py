# ============================================================================
# analysis/views.py
# Plant Disease Detection — Django Views (Optimized)
# ============================================================================

import json
import os
import re
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .forms import SignUpForm
from .ml_utils import (
    ai_compare_evolution,
    ai_doctor_report,
    ai_gemini_doctor,
    ai_llm_doctor,
    ai_treatment_advisor,
    classifier_instance,
    draw_red_regions_boxes,
    estimate_disease_progress,
)
from .models import CustomUser, FollowUpScan, ScanResult


# ============================================================================
# Main Dashboard / Home
# ============================================================================

def home(request):
    """Main diagnosis dashboard — upload, predict, and analyze plant images."""
    context = {
        "result": None,
        "image_url": None,
        "gradcam_url": None,
        "progress_ratio": 0,
        "progress_stage": "Unknown",
    }

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]

        # Prediction
        image_file.seek(0)
        result = classifier_instance.predict(image_file)

        disease = result["disease"]
        confidence = result["confidence"]

        # Parse confidence to float
        if isinstance(confidence, str):
            confidence = float(confidence.replace("%", ""))

        context["result"] = result

        fs = FileSystemStorage()
        uid = uuid.uuid4().hex[:8]
        clean_name = re.sub(r"[^a-zA-Z0-9_.]", "_", image_file.name)

        original_name = f"{uid}_orig_{clean_name}"
        original_path = fs.save(original_name, image_file)
        context["image_url"] = fs.url(original_path)

        # Grad-CAM
        image_file.seek(0)
        gradcam = classifier_instance.apply_gradcam(image_file)

        if gradcam:
            boxed = draw_red_regions_boxes(gradcam["superimposed"])

            gradcam_name = f"{uid}_gradcam.png"
            gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_name)
            boxed.save(gradcam_path, format="PNG")
            context["gradcam_url"] = settings.MEDIA_URL + gradcam_name

            # Disease progress
            ratio, stage = estimate_disease_progress(gradcam["superimposed"])
            context["progress_ratio"] = ratio
            context["progress_stage"] = stage

            # AI treatment advisor
            recommendations = ai_treatment_advisor(disease, confidence, stage)
            context["result"]["recommendations"] = recommendations

            # AI LLM doctor
            llm_text = ai_llm_doctor(disease, confidence, stage, ratio)
            context["result"]["ai_doctor"] = llm_text

            # AI Gemini doctor
            gemini_text = ai_gemini_doctor(disease, confidence, stage, ratio)
            context["ai_report"] = gemini_text

    return render(request, "analysis/dashboard.html", context)


# ============================================================================
# Upload (AJAX endpoint)
# ============================================================================

def upload_image(request):
    """AJAX endpoint for image upload and prediction."""
    context = {
        "result": None,
        "image_url": None,
        "gradcam_url": None,
        "error": None,
    }

    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        fs = FileSystemStorage()

        try:
            # Prediction
            image_file.seek(0)
            result = classifier_instance.predict(image_file)
            context["result"] = result

            # Save original image
            unique_id = uuid.uuid4().hex[:8]
            original_filename = f"{unique_id}_original_{image_file.name}"
            original_path = fs.save(original_filename, image_file)
            context["image_url"] = fs.url(original_path)

            # Grad-CAM
            image_file.seek(0)
            gradcam_result = classifier_instance.apply_gradcam(image_file)

            if gradcam_result:
                gradcam_filename = f"{unique_id}_gradcam.png"
                gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_filename)
                gradcam_result["superimposed"].save(gradcam_path, format="PNG")
                context["gradcam_url"] = settings.MEDIA_URL + gradcam_filename
                context["gradcam_class"] = gradcam_result["predicted_class"].replace("_", " ")

        except Exception as e:
            import traceback
            traceback.print_exc()
            context["error"] = str(e)

        return JsonResponse(context)

    return render(request, "analysis/upload.html", context)


# ============================================================================
# Static Pages
# ============================================================================

def model_info(request):
    """Model information page."""
    return render(request, "analysis/model.html")


def history(request):
    """Analysis history page."""
    return render(request, "analysis/history.html")


# ============================================================================
# Authentication
# ============================================================================

def login_view(request):
    """User login page."""
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect("home")
    else:
        form = AuthenticationForm()
    return render(request, "analysis/login.html", {"form": form})


def signup_view(request):
    """User registration page."""
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created successfully!")
            return redirect("home")
    else:
        form = SignUpForm()
    return render(request, "analysis/signup.html", {"form": form})


# ============================================================================
# Progress View
# ============================================================================

def progress_view(request):
    """Detailed disease progression analysis page."""
    context = {
        "gradcam_url": request.GET.get("img"),
        "ratio": request.GET.get("ratio"),
        "stage": request.GET.get("stage"),
    }
    return render(request, "analysis/progress.html", context)


# ============================================================================
# Authentication
# ============================================================================

def logout_view(request):
    """Log the user out and redirect to login."""
    logout(request)
    return redirect("home")


# ============================================================================
# Save Scan
# ============================================================================

def save_scan(request):
    """Save a scan result via AJAX POST."""
    if request.method == "POST":
        photo_id = uuid.uuid4().hex[:12]

        orig_path = request.POST.get("orig", "")
        grad_path = request.POST.get("gradcam", "")

        try:
            ScanResult.objects.create(
                photo_id=photo_id,
                user=request.user,
                image_original=orig_path.replace("/media/", ""),
                image_gradcam=grad_path.replace("/media/", ""),
                prediction=request.POST.get("prediction", ""),
                confidence=float(request.POST.get("confidence", "0").replace("%", "")),
                disease_ratio=float(request.POST.get("ratio", "0")),
                disease_stage=request.POST.get("stage", "Unknown"),
                ai_medical=request.POST.get("ai_medical", ""),
                ai_treatment=request.POST.get("ai_treatment", ""),
                ai_irrigation=request.POST.get("ai_irrigation", ""),
                ai_economic=request.POST.get("ai_economic", ""),
                yield_loss=float(request.POST.get("yield_loss", "0")),
                fungicides_json=request.POST.get("fungicides_json", "[]"),
                folder_name=request.POST.get("folder_name", "Untitled"),
            )
            messages.success(request, "Scan saved successfully")
            return JsonResponse({"status": "ok"})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)

    return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)


# ============================================================================
# My Files
# ============================================================================

def my_files(request):
    """Display all saved scans for the current user."""
    scans = ScanResult.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "analysis/my_files.html", {"scans": scans})


# ============================================================================
# Scan Detail
# ============================================================================

def scan_detail(request, photo_id):
    """Detailed view of a single saved scan with follow-up history."""
    scan = get_object_or_404(ScanResult, photo_id=photo_id, user=request.user)
    progress = scan.progress_scans.all().order_by("-created_at")

    return render(
        request,
        "analysis/scan_detail.html",
        {"scan": scan, "progress": progress},
    )


# ============================================================================
# Compare Scan
# ============================================================================

def compare_scan(request, photo_id):
    """Compare a new scan against a previous one."""
    scan = get_object_or_404(ScanResult, photo_id=photo_id)

    context = {
        "scan": scan,
        "new_result": None,
        "new_gradcam": None,
        "new_ratio": None,
        "new_stage": None,
        "comparison_text": None,
    }

    if request.method == "POST" and request.FILES.get("new_image"):
        image_file = request.FILES["new_image"]

        image_file.seek(0)
        result = classifier_instance.predict(image_file)

        fs = FileSystemStorage()
        uid = uuid.uuid4().hex[:8]
        clean_name = re.sub(r"[^a-zA-Z0-9_.]", "_", image_file.name)

        original_name = f"{uid}_compare_{clean_name}"
        original_path = fs.save(original_name, image_file)
        context["new_image_url"] = fs.url(original_path)

        # Grad-CAM
        image_file.seek(0)
        gradcam = classifier_instance.apply_gradcam(image_file)

        if gradcam:
            boxed = draw_red_regions_boxes(gradcam["superimposed"])

            gradcam_name = f"{uid}_compare_gradcam.png"
            gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_name)
            boxed.save(gradcam_path, format="PNG")
            context["new_gradcam"] = settings.MEDIA_URL + gradcam_name

            ratio, stage = estimate_disease_progress(gradcam["superimposed"])
            context["new_ratio"] = ratio
            context["new_stage"] = stage

            # Comparison logic
            old_ratio = float(scan.disease_ratio)
            diff = ratio - old_ratio

            if diff > 5:
                text = f"Disease increased by {round(diff, 2)}%. Infection spreading."
            elif diff < -5:
                text = f"Disease decreased by {abs(round(diff, 2))}%. Plant recovering."
            else:
                text = "Disease level stable."

            context["comparison_text"] = text

        context["new_result"] = result

    return render(request, "analysis/scan_detail.html", context)


# ============================================================================
# Compare AJAX
# ============================================================================

def compare_ajax(request, photo_id):
    """AJAX endpoint for comparing disease evolution."""
    if request.method != "POST" or not request.FILES.get("new_image"):
        return JsonResponse(
            {"error": "Invalid request method or missing image"}, status=400
        )

    scan = get_object_or_404(ScanResult, photo_id=photo_id)
    image_file = request.FILES["new_image"]

    # Prediction
    image_file.seek(0)
    result = classifier_instance.predict(image_file)

    # Grad-CAM
    image_file.seek(0)
    gradcam = classifier_instance.apply_gradcam(image_file)

    # Save Grad-CAM image
    filename = f"{uuid.uuid4().hex}_compare_gradcam.png"
    filepath = os.path.join(settings.MEDIA_ROOT, filename)

    heat = gradcam["superimposed"]
    try:
        heat.save(filepath, format="PNG")
    except AttributeError:
        from PIL import Image
        Image.fromarray(heat).save(filepath)

    gradcam_url = settings.MEDIA_URL + filename

    # Save original new image
    filename2 = f"{uuid.uuid4().hex}_compare_original.jpg"
    filepath2 = os.path.join(settings.MEDIA_ROOT, filename2)
    with open(filepath2, "wb+") as f:
        for chunk in image_file.chunks():
            f.write(chunk)
    original_url = settings.MEDIA_URL + filename2

    # Disease progress
    ratio, stage = estimate_disease_progress(gradcam["superimposed"])

    # AI evolution analysis
    evolution = ai_compare_evolution(scan.prediction, scan.disease_ratio, ratio)

    # AI doctor report
    ai = ai_doctor_report(result["disease"], ratio)

    return JsonResponse({
        "status": "ok",
        "gradcam": gradcam_url,
        "original": original_url,
        "disease": result["disease"],
        "confidence": result["confidence"],
        "ratio": ratio,
        "stage": stage,
        "medical": ai["medical"],
        "treatment": ai["treatment"],
        "irrigation": ai["irrigation"],
        "economic": ai["economic_risk"],
        "yield": ai["yield_loss_percent"],
        "fungicides": ai["fungicides"],
        "evolution": evolution,
    })


# ============================================================================
# Save Follow-Up
# ============================================================================

def save_followup(request, photo_id):
    """Save a follow-up comparison scan."""
    if request.method != "POST":
        return JsonResponse({"status": "error"}, status=400)

    try:
        data = json.loads(request.body)

        parent = get_object_or_404(ScanResult, photo_id=photo_id, user=request.user)

        FollowUpScan.objects.create(
            follow_id=uuid.uuid4().hex,
            parent_scan=parent,
            user=request.user,
            new_image=data["image"],
            new_gradcam=data["gradcam"],
            disease=data["disease"],
            confidence=float(data["confidence"].replace("%", "")),
            ratio=data["ratio"],
            stage=data["stage"],
            yield_loss=data["yield"],
            evolution_text=data["evolution"],
        )

        return JsonResponse({"status": "saved"})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)


# ============================================================================
# Treatment View
# ============================================================================

def treatment_view(request):
    """AI-powered treatment recommendations based on disease analysis."""
    stage = request.GET.get("stage", "Unknown")
    ratio = request.GET.get("ratio", "0")
    disease = request.GET.get("disease", "")

    try:
        ratio_val = float(ratio)
    except (ValueError, TypeError):
        ratio_val = 0.0

    # Treatment logic based on severity
    if ratio_val >= 70:
        urgency = "critical"
        actions = [
            "Isolate affected plants immediately",
            "Remove severely infected leaves/plants",
            "Apply systemic fungicide within 24 hours",
            "Consult agricultural specialist",
            "Monitor neighboring plants daily",
        ]
    elif ratio_val >= 40:
        urgency = "moderate"
        actions = [
            "Apply targeted fungicide treatment",
            "Improve air circulation around plants",
            "Adjust irrigation to reduce leaf wetness",
            "Schedule follow-up scan in 3-5 days",
            "Sanitize tools after handling",
        ]
    else:
        urgency = "early"
        actions = [
            "Monitor plant daily for changes",
            "Apply preventive bio-fungicide",
            "Ensure optimal light and nutrition",
            "Document progress with photos",
            "Re-scan if symptoms worsen",
        ]

    fungicides = [
        {"name": "Azoxystrobin", "type": "Systemic", "dosage": "0.5-1.0 L/ha"},
        {"name": "Copper Oxychloride", "type": "Contact", "dosage": "2-3 kg/ha"},
        {"name": "Bacillus subtilis", "type": "Biological", "dosage": "1-2 L/ha"},
    ]

    context = {
        "stage": stage,
        "ratio": ratio_val,
        "disease": disease,
        "urgency": urgency,
        "actions": actions,
        "fungicides": fungicides,
        "back_url": request.META.get("HTTP_REFERER", "/"),
    }

    return render(request, "analysis/treatment.html", context)