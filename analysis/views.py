# ============================================================================
# analysis/views.py
# Plant Disease Detection — Django Views (Optimized)
# ============================================================================

import json
import os
import re
import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from .services.ai_service import get_gemini_service
from django.utils import timezone
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
from .models import CustomUser, FollowUpScan, ScanResult, FakePayment, Complaint, PLAN_CONFIG, ChatSession, ChatMessage
from .decorators import scan_limit_required, premium_required
from .services.subscription import check_scan_limit, record_scan, get_all_plans, upgrade_user
from django.views.decorators.http import require_POST


# ============================================================================
# Main Dashboard / Home
# ============================================================================

def home(request):
    """Main diagnosis dashboard — upload, predict, and analyze plant images."""
    if not request.user.is_authenticated:
        return redirect("login")

    context = {
        "result": None,
        "image_url": None,
        "gradcam_url": None,
        "progress_ratio": 0,
        "progress_stage": "Unknown",
    }

    # ── Restore last analysis from session (Issue #1: persistence) ──
    if request.method == "GET":
        last_analysis = request.session.get("last_analysis")
        if last_analysis:
            context["result"] = last_analysis.get("result")
            context["image_url"] = last_analysis.get("image_url")
            context["gradcam_url"] = last_analysis.get("gradcam_url")
            context["progress_ratio"] = last_analysis.get("progress_ratio", 0)
            context["progress_stage"] = last_analysis.get("progress_stage", "Unknown")
            context["ai_report"] = last_analysis.get("ai_report")
        return render(request, "analysis/dashboard.html", context)

    if request.method == "POST" and request.FILES.get("image"):
        # ── Scan Limit Check ──
        limit_result = check_scan_limit(request.user)
        if not limit_result["allowed"]:
            messages.warning(request, limit_result["message"])
            return render(request, "analysis/dashboard.html", context)

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

        # ── Record Scan ──
        record_scan(request.user)

        # ── Store last analysis in session for persistence (Issue #1) ──
        try:
            request.session["last_analysis"] = {
                "result": context.get("result"),
                "image_url": context.get("image_url"),
                "gradcam_url": context.get("gradcam_url"),
                "progress_ratio": context.get("progress_ratio", 0),
                "progress_stage": context.get("progress_stage", "Unknown"),
                "ai_report": context.get("ai_report"),
            }
        except Exception:
            pass  # Session serialization failure is non-critical

    return render(request, "analysis/dashboard.html", context)


@login_required
@require_POST
def clear_last_analysis(request):
    """AJAX endpoint to clear the last analysis from session."""
    request.session.pop("last_analysis", None)
    return JsonResponse({"status": "ok"})

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
        # ── Scan Limit Check ──
        if request.user.is_authenticated:
            limit_result = check_scan_limit(request.user)
            if not limit_result["allowed"]:
                return JsonResponse({
                    "error": limit_result["message"],
                    "limit_reached": True,
                    "plan_type": limit_result["plan_type"],
                }, status=403)

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

            # ── Record Scan ──
            if request.user.is_authenticated:
                record_scan(request.user)

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


# history view removed — feature #8


# ============================================================================
# Authentication
# ============================================================================

def login_view(request):
    """User login page."""
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            if not user.email_verified and not user.is_staff:
                # Redirect to 6-digit code verification page
                from .services.email_service import send_verification_email
                send_verification_email(user, request=request)
                return render(request, "analysis/email_verify.html", {
                    "email": user.email,
                    "user_pk": user.pk,
                })
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect("home")
    else:
        form = AuthenticationForm()
    return render(request, "analysis/login.html", {"form": form})


def signup_view(request):
    """User registration with 6-digit email verification code."""
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email_verified = False
            user.save()

            # ── Auto-create CompanyProfile for company accounts ──────
            if getattr(user, "purpose", None) == "company":
                try:
                    from companies.models import CompanyProfile
                    CompanyProfile.objects.get_or_create(
                        user=user,
                        defaults={
                            "company_name": user.username,
                            "email": user.email,
                            "is_active": True,
                        },
                    )
                    user.role = "company"
                    user.save(update_fields=["role"])
                except Exception:
                    pass  # Non-critical; user can complete profile via company dashboard

            # Send 6-digit verification code
            from .services.email_service import send_verification_email
            send_verification_email(user, request=request)

            return render(request, "analysis/email_verify.html", {
                "email": user.email,
                "user_pk": user.pk,
            })
    else:
        form = SignUpForm()
    return render(request, "analysis/signup.html", {"form": form})



def verify_code_view(request):
    """Handle 6-digit email verification code submission."""
    if request.method == "POST":
        user_pk = request.POST.get("user_pk")
        code = request.POST.get("code", "").strip()

        if not code or len(code) != 6:
            messages.error(request, "Please enter a valid 6-digit code.")
            return render(request, "analysis/email_verify.html", {
                "user_pk": user_pk,
                "email": request.POST.get("email", ""),
                "error": "Please enter a valid 6-digit code.",
            })

        from .services.email_service import verify_code

        success, result = verify_code(user_pk, code)

        if success:
            messages.success(request, "Email verified successfully! You can now log in.")
            return render(request, "analysis/email_verified.html", {"user": result})
        else:
            # Get user email for the form
            email = ""
            try:
                user = CustomUser.objects.get(pk=user_pk)
                email = user.email
            except CustomUser.DoesNotExist:
                pass

            messages.error(request, result)
            return render(request, "analysis/email_verify.html", {
                "user_pk": user_pk,
                "email": email,
                "error": result,
            })

    return redirect("login")


def verify_email_view(request, token):
    """Legacy link-based verification — redirect to code-based flow."""
    messages.info(request, "Please use the 6-digit verification code sent to your email.")
    return redirect("login")


def resend_verification(request):
    """Resend 6-digit verification code."""
    if request.method == "POST":
        user_pk = request.POST.get("user_pk")
        try:
            user = CustomUser.objects.get(pk=user_pk)
            if not user.email_verified:
                from .services.email_service import send_verification_email
                send_verification_email(user, request=request)
                messages.success(request, "New verification code sent! Check your inbox.")
                return render(request, "analysis/email_verify.html", {
                    "email": user.email,
                    "user_pk": user.pk,
                })
            else:
                messages.info(request, "Email already verified.")
        except CustomUser.DoesNotExist:
            messages.error(request, "User not found.")
    return redirect("login")


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


@login_required
@require_POST
def delete_scan_file(request, photo_id):
    """Delete a scan from My Files — removes images + database record."""
    scan = get_object_or_404(ScanResult, photo_id=photo_id, user=request.user)

    # Delete physical files from disk
    try:
        if scan.image_original:
            file_path = os.path.join(settings.MEDIA_ROOT, str(scan.image_original))
            if os.path.exists(file_path):
                os.remove(file_path)
        if scan.image_gradcam:
            gradcam_path = os.path.join(settings.MEDIA_ROOT, str(scan.image_gradcam))
            if os.path.exists(gradcam_path):
                os.remove(gradcam_path)
    except Exception:
        pass  # File cleanup failure is non-critical

    # Delete follow-up scans
    scan.progress_scans.all().delete()

    # Delete database record
    prediction = scan.prediction
    scan.delete()

    messages.success(request, f"Scan '{prediction}' deleted successfully.")
    return redirect("my_files")


# ============================================================================
# Scan Detail
# ============================================================================
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from .models import FollowUpScan

@csrf_exempt
def save_followupnew(request, photo_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)

    scan = get_object_or_404(ScanResult, photo_id=photo_id, user=request.user)

    data = json.loads(request.body)

    follow = FollowUpScan.objects.create(
        follow_id=str(uuid.uuid4())[:12],
        user=request.user,
        parent_scan=scan,

        disease=data.get("disease"),
        confidence=float(str(data.get("confidence")).replace("%", "")),
        ratio=data.get("ratio"),
        stage=data.get("stage"),
        yield_loss=data.get("yield", 0),

        evolution_text=data.get("evolution"),

        # ⚠️ صور (اختياري)
        new_image=data.get("original"),
        new_gradcam=data.get("gradcam"),
        ai_medical=data.get("ai_medical"),
        ai_treatment=data.get("ai_treatment"),
        ai_irrigation=data.get("ai_irrigation"),
        ai_economic=data.get("ai_economic"),
    )
    print("🔥 SAVE CALLED", data)
    return JsonResponse({"status": "saved"})

def scan_detail(request, photo_id):
    """Detailed view of a single saved scan with follow-up history."""
    scan = get_object_or_404(ScanResult, photo_id=photo_id, user=request.user)

    progress = scan.progress_scans.all().order_by("-created_at")

    return render(
        request,
        "analysis/scan_detail.html",
        {
            "scan": scan,
            "progress": progress,
        },
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

from .services.ai_service import get_gemini_service

def compare_ajax(request, photo_id):
    """AJAX endpoint for comparing disease evolution."""
    if request.method != "POST" or not request.FILES.get("new_image"):
        return JsonResponse(
            {"error": "Invalid request method or missing image"}, status=400
        )

    # ── Scan Limit Check ──
    if request.user.is_authenticated:
        limit_result = check_scan_limit(request.user)
        if not limit_result["allowed"]:
            return JsonResponse({
                "error": limit_result["message"],
                "limit_reached": True,
            }, status=403)

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

    # AI doctor report (old system)
    ai = ai_doctor_report(result["disease"], ratio)

    # ── Gemini AI — with automatic Local AI fallback ──────────────────
    gemini = get_gemini_service()

    ai_medical = ""
    ai_treatment = ""
    ai_irrigation = ""
    ai_economic = ""

    if gemini and gemini.is_available:
        scan_data = {
            "disease": result["disease"],
            "confidence": result["confidence"],
            "ratio": ratio,
            "stage": stage,
            "yield_loss": ai.get("yield_loss_percent", 0)
        }

        try:
            ai_medical = gemini.summarize_scan(scan_data).get("text", "")
            ai_treatment = gemini.generate_treatment_plan(
                result["disease"], stage, ratio
            ).get("text", "")
            ai_irrigation = gemini.analyze_text(
                f"Irrigation advice for {result['disease']} at {ratio}% severity"
            ).get("text", "")
            ai_economic = gemini.analyze_text(
                f"Economic impact of {result['disease']} with {ratio}% severity"
            ).get("text", "")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Gemini error in compare_ajax: %s", e)

    # ── Local AI fallback: fill any blank fields ──────────────────────
    if not all([ai_medical, ai_treatment, ai_irrigation, ai_economic]):
        local = ai
        if not ai_medical:
            ai_medical = local["medical"]
        if not ai_treatment:
            ai_treatment = local["treatment"]
        if not ai_irrigation:
            ai_irrigation = local["irrigation"]
        if not ai_economic:
            ai_economic = local["economic_risk"]

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

        # AI Report fields — always populated (Gemini or Local AI)
        "ai_medical": ai_medical,
        "ai_treatment": ai_treatment,
        "ai_irrigation": ai_irrigation,
        "ai_economic": ai_economic,

        "created_at": timezone.now().strftime("%Y-%m-%d %H:%M")
    })

    # ── Record Scan after successful comparison ──
    if request.user.is_authenticated:
        record_scan(request.user)


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


# ============================================================================
# AI Assistant Page
# ============================================================================

@login_required
def assistant_view(request):
    """AI Assistant chat interface — Premium only, multi-session."""
    if not request.user.can_use_chat():
        return render(request, "analysis/assistant.html", {
            "chat_locked": True,
        })

    # Get user's chat sessions
    sessions = ChatSession.objects.filter(user=request.user).order_by("-updated_at")[:50]

    # Pass user's recent scans for "Attach Scan" feature
    user_scans = ScanResult.objects.filter(user=request.user).order_by("-created_at")[:20]

    return render(request, "analysis/assistant.html", {
        "chat_locked": False,
        "user_scans": user_scans,
        "chat_sessions": sessions,
    })


# ============================================================================
# AI API Endpoints (Gemini-powered)
# ============================================================================

@login_required
@premium_required
def ai_chat(request):
    """POST /api/ai/chat/ — Send a message through the Hybrid AI pipeline."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()
        context = data.get("context", [])
        scan_id = data.get("scan_id", None)  # Optional scan attachment

        if not message:
            return JsonResponse({"error": "Message cannot be empty"}, status=400)

        if len(message) > 2000:
            return JsonResponse({"error": "Message too long (max 2000 chars)"}, status=400)

        # Build scan context if scan_id is provided
        scan_context = None
        if scan_id:
            try:
                scan = ScanResult.objects.get(photo_id=scan_id, user=request.user)
                scan_context = {
                    "disease": scan.prediction,
                    "confidence": str(scan.confidence),
                    "ratio": str(scan.disease_ratio),
                    "stage": scan.disease_stage,
                    "yield_loss": str(scan.yield_loss) if scan.yield_loss else "N/A",
                    "recommendations": scan.ai_treatment or "",
                }
            except ScanResult.DoesNotExist:
                pass  # Proceed without scan context

        # Detect language from message or Django locale
        language = data.get("language") or getattr(request, 'LANGUAGE_CODE', 'en')

        # Use Hybrid AI Service (Gemini + Local fallback)
        from .services.hybrid_ai_service import get_hybrid_ai_service
        service = get_hybrid_ai_service()

        result = service.chat(
            message=message,
            scan_context=scan_context,
            chat_context=context,
            language=language,
        )

        if "error" in result and result.get("status") == "error":
            return JsonResponse({"error": result.get("text", "AI service error")}, status=500)

        return JsonResponse({
            "status": "ok",
            "response": result["text"],
            "offline": result.get("offline", False),
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)


@login_required
@premium_required
def ai_summarize(request, photo_id):
    """POST /api/ai/summarize/<photo_id>/ — Summarize a specific scan."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        scan = get_object_or_404(ScanResult, photo_id=photo_id, user=request.user)

        scan_data = {
            "disease": scan.prediction,
            "confidence": str(scan.confidence),
            "ratio": str(scan.disease_ratio),
            "stage": scan.disease_stage,
            "yield_loss": str(scan.yield_loss),
        }

        from .services.ai_service import get_gemini_service
        service = get_gemini_service()

        if not service.is_available:
            return JsonResponse({"error": "AI service unavailable"}, status=503)

        result = service.summarize_scan(scan_data)

        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=500)

        return JsonResponse({
            "status": "ok",
            "summary": result["text"],
            "scan": scan_data,
        })

    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)


@login_required
@premium_required
def ai_treatment_plan(request):
    """POST /api/ai/treatment-plan/ — Generate a detailed treatment plan."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        disease = data.get("disease", "").strip()
        stage = data.get("stage", "").strip()
        ratio = data.get("ratio", "0")

        if not disease:
            return JsonResponse({"error": "Disease name is required"}, status=400)

        from .services.ai_service import get_gemini_service
        service = get_gemini_service()

        if not service.is_available:
            return JsonResponse({"error": "AI service unavailable"}, status=503)

        result = service.generate_treatment_plan(disease, stage, ratio)

        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=500)

        return JsonResponse({
            "status": "ok",
            "plan": result["text"],
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)


@login_required
@premium_required
def ai_recommendations(request):
    """GET /api/ai/recommendations/ — Smart recommendations based on scan history."""
    try:
        scans = ScanResult.objects.filter(user=request.user).order_by("-created_at")[:10]

        if not scans:
            return JsonResponse({
                "status": "ok",
                "recommendations": "No scan history found. Upload your first plant image to get AI-powered recommendations!",
            })

        scan_history = [
            {
                "disease": s.prediction,
                "ratio": str(s.disease_ratio),
                "stage": s.disease_stage,
                "date": s.created_at.strftime("%Y-%m-%d"),
            }
            for s in scans
        ]

        from .services.ai_service import get_gemini_service
        service = get_gemini_service()

        if not service.is_available:
            return JsonResponse({"error": "AI service unavailable"}, status=503)

        result = service.get_recommendations(scan_history)

        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=500)

        return JsonResponse({
            "status": "ok",
            "recommendations": result["text"],
            "scan_count": len(scan_history),
        })

    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)



@csrf_exempt
def save_followupp(request, photo_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=400)

    scan = get_object_or_404(ScanResult, photo_id=photo_id)

    data = json.loads(request.body)

    follow = FollowUpScan.objects.create(
        scan=scan,
        prediction=data.get("disease"),
        confidence=data.get("confidence"),
        disease_ratio=data.get("ratio"),
        disease_stage=data.get("stage"),
        ai_medical=data.get("ai_medical"),
        ai_treatment=data.get("ai_treatment"),
        ai_irrigation=data.get("ai_irrigation"),
        ai_economic=data.get("ai_economic"),
        evolution=data.get("evolution"),
    )

    return JsonResponse({"status": "saved"})


# ============================================================================
# Subscription Status API
# ============================================================================

@login_required
def subscription_status(request):
    """GET /api/subscription/status/ — Return current user's subscription info."""
    user = request.user
    remaining = user.remaining_scans_today()
    limit = user.get_daily_limit()

    return JsonResponse({
        "status": "ok",
        "plan_type": user.plan_type,
        "plan_label": user.plan_config["label"],
        "remaining_scans": remaining,
        "daily_limit": limit,
        "daily_scan_count": user.daily_scan_count,
        "total_scans": user.total_scans_count,
        "can_chat": user.can_use_chat(),
        "is_premium": user.plan_type == "premium",
        "subscription_active": user.is_subscription_active(),
        "subscription_end": user.subscription_end.isoformat() if user.subscription_end else None,
    })


# ============================================================================
# Plans / Pricing Page
# ============================================================================

@login_required
def plans_page(request):
    """Display pricing plans with feature comparison."""
    plans = get_all_plans()
    return render(request, "analysis/plans.html", {
        "plans": plans,
    })


# ============================================================================
# Upgrade Plan (Fake Payment)
# ============================================================================

@login_required
def upgrade_plan(request):
    """Fake payment form for upgrading subscription plan (demo mode)."""
    selected_plan = request.GET.get("plan", "basic")
    if selected_plan not in ("basic", "premium"):
        selected_plan = "basic"

    plan_config = PLAN_CONFIG.get(selected_plan, PLAN_CONFIG["basic"])

    context = {
        "selected_plan": selected_plan,
        "plan_config": plan_config,
        "errors": {},
        "form_data": {},
    }

    if request.method == "POST":
        full_name = request.POST.get("full_name", "").strip()
        wallet_number = request.POST.get("wallet_number", "").strip()
        plan = request.POST.get("selected_plan", "").strip()

        context["form_data"] = {
            "full_name": full_name,
            "wallet_number": wallet_number,
        }

        errors = {}

        # Validation
        if not full_name:
            errors["full_name"] = "Full name is required."
        elif len(full_name) < 2:
            errors["full_name"] = "Full name must be at least 2 characters."

        if not wallet_number:
            errors["wallet_number"] = "Wallet number is required."
        elif not wallet_number.isdigit():
            errors["wallet_number"] = "Wallet number must contain only digits."
        elif len(wallet_number) < 6 or len(wallet_number) > 20:
            errors["wallet_number"] = "Wallet number must be 6–20 digits."

        if plan not in ("basic", "premium"):
            errors["selected_plan"] = "Invalid plan selected."
        else:
            selected_plan = plan
            plan_config = PLAN_CONFIG.get(selected_plan, PLAN_CONFIG["basic"])
            context["selected_plan"] = selected_plan
            context["plan_config"] = plan_config

        if errors:
            context["errors"] = errors
            return render(request, "analysis/upgrade.html", context)

        # Check if user already on this plan
        if request.user.plan_type == selected_plan and request.user.is_subscription_active():
            messages.info(request, f"You are already on the {plan_config['label']} plan.")
            return render(request, "analysis/upgrade.html", context)

        # Create FakePayment record
        payment = FakePayment.objects.create(
            user=request.user,
            full_name=full_name,
            wallet_number=wallet_number,
            plan_requested=selected_plan,
            amount=plan_config["price"],
            status="pending",
        )

        # Auto-approve immediately
        payment.status = "approved"
        payment.approved_at = timezone.now()
        payment.save(update_fields=["status", "approved_at"])

        # Upgrade user
        success, msg = upgrade_user(request.user, selected_plan, duration_days=30)

        if success:
            # Auto-assign experts when upgrading to premium
            if selected_plan == "premium":
                try:
                    from experts.auto_linking import auto_assign_experts_to_user
                    auto_assign_experts_to_user(request.user)
                except Exception:
                    pass  # Non-critical — don't block upgrade

            messages.success(
                request,
                f"Your plan has been upgraded to {plan_config['label']} successfully! "
                f"Your subscription is active until {request.user.subscription_end.strftime('%B %d, %Y')}."
            )
            return redirect("home")
        else:
            messages.error(request, f"Upgrade failed: {msg}")
            return render(request, "analysis/upgrade.html", context)

    return render(request, "analysis/upgrade.html", context)


# ============================================================================
# Settings Page
# ============================================================================

@login_required
def settings_page(request):
    """User settings page with 3 tabs: Info, Edit, Statistics."""
    user = request.user
    tab = request.GET.get("tab", "info")
    errors = {}
    success = False

    if request.method == "POST":
        tab = "edit"
        action = request.POST.get("action", "")

        if action == "update_profile":
            first_name = request.POST.get("first_name", "").strip()
            last_name = request.POST.get("last_name", "").strip()
            email = request.POST.get("email", "").strip()

            if not email:
                errors["email"] = "Email is required."
            elif CustomUser.objects.filter(email=email).exclude(pk=user.pk).exists():
                errors["email"] = "This email is already in use."

            if not errors:
                user.first_name = first_name
                user.last_name = last_name
                user.email = email
                user.save(update_fields=["first_name", "last_name", "email"])
                messages.success(request, "Profile updated successfully.")
                success = True

        elif action == "change_password":
            current = request.POST.get("current_password", "")
            new_pw = request.POST.get("new_password", "")
            confirm = request.POST.get("confirm_password", "")

            if not user.check_password(current):
                errors["current_password"] = "Current password is incorrect."
            elif len(new_pw) < 6:
                errors["new_password"] = "Password must be at least 6 characters."
            elif new_pw != confirm:
                errors["confirm_password"] = "Passwords do not match."

            if not errors:
                user.set_password(new_pw)
                user.save()
                from django.contrib.auth import update_session_auth_hash
                update_session_auth_hash(request, user)
                messages.success(request, "Password changed successfully.")
                success = True

    # Stats tab data
    today = timezone.localdate()
    seven_days_ago = today - timedelta(days=7)

    stats = {
        "total_scans": ScanResult.objects.filter(user=user).count(),
        "scans_today": ScanResult.objects.filter(user=user, created_at__date=today).count(),
        "scans_week": ScanResult.objects.filter(user=user, created_at__date__gte=seven_days_ago).count(),
        "plan_type": user.effective_plan_type,
        "plan_label": user.plan_config["label"],
        "subscription_end": user.subscription_end,
    }

    # Payment / subscription history
    payments = FakePayment.objects.filter(user=user, status="approved").order_by("-created_at")[:20]

    context = {
        "tab": tab,
        "errors": errors,
        "success": success,
        "user_stats": stats,
        "payments": payments,
    }
    return render(request, "analysis/settings.html", context)


# ============================================================================
# Complaints (Shkawi) — User Side
# ============================================================================

@login_required
def my_complaints(request):
    """User complaint list + submit new complaint."""
    # Admin cannot submit complaints (#13)
    if request.user.is_staff:
        messages.info(request, "Administrators cannot submit complaints.")
        return redirect("home")

    errors = {}

    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        message_text = request.POST.get("message", "").strip()

        if not title:
            errors["title"] = "Title is required."
        elif len(title) < 3:
            errors["title"] = "Title must be at least 3 characters."

        if not message_text:
            errors["message"] = "Message is required."
        elif len(message_text) < 10:
            errors["message"] = "Message must be at least 10 characters."

        if not errors:
            Complaint.objects.create(
                user=request.user,
                title=title,
                message=message_text,
            )
            messages.success(request, "Your complaint has been submitted successfully.")
            return redirect("my_complaints")

    complaints = Complaint.objects.filter(user=request.user)

    context = {
        "complaints": complaints,
        "errors": errors,
    }
    return render(request, "analysis/complaints.html", context)


@login_required
def complaint_detail(request, complaint_id):
    """View a single complaint detail (user-side)."""
    complaint = get_object_or_404(Complaint, pk=complaint_id, user=request.user)
    return render(request, "analysis/complaint_detail.html", {"complaint": complaint})


@login_required
@require_POST
def delete_complaint(request, complaint_id):
    """Delete a user's own complaint."""
    complaint = get_object_or_404(Complaint, pk=complaint_id, user=request.user)
    complaint.delete()
    messages.success(request, "Complaint deleted successfully.")
    return redirect("my_complaints")


# ============================================================================
# Chat Session API Views (Multi-Conversation System)
# ============================================================================

@login_required
@premium_required
def chat_sessions_list(request):
    """GET — Return all chat sessions for the current user."""
    sessions = ChatSession.objects.filter(user=request.user).order_by("-updated_at")[:50]
    data = [
        {
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.strftime("%Y-%m-%d %H:%M"),
            "updated_at": s.updated_at.strftime("%Y-%m-%d %H:%M"),
            "message_count": s.messages.count(),
        }
        for s in sessions
    ]
    return JsonResponse({"status": "ok", "sessions": data})


@login_required
@premium_required
def chat_session_create(request):
    """POST — Create a new chat session. Reuses empty sessions to prevent duplicates."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    # ── Prevent duplicate empty sessions ─────────────────────────────
    # If there's already an active session with no messages, reuse it
    empty_active = (
        ChatSession.objects
        .filter(user=request.user, is_active=True, message_count=0)
        .first()
    )
    if empty_active:
        return JsonResponse({
            "status": "ok",
            "session": {
                "id": empty_active.id,
                "title": empty_active.title,
                "created_at": empty_active.created_at.strftime("%Y-%m-%d %H:%M"),
            },
        })

    # Deactivate any previously active sessions (keep them accessible in history)
    ChatSession.objects.filter(
        user=request.user, is_active=True
    ).update(is_active=False)

    session = ChatSession.objects.create(
        user=request.user,
        title="New Chat",
        is_active=True,
        is_closed=False,
    )
    return JsonResponse({
        "status": "ok",
        "session": {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.strftime("%Y-%m-%d %H:%M"),
        },
    })


@login_required
@premium_required
def chat_session_messages(request, session_id):
    """GET — Return messages for a specific session."""
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    msgs = session.messages.order_by("created_at")
    data = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "is_offline": m.is_offline,
            "created_at": m.created_at.strftime("%H:%M"),
        }
        for m in msgs
    ]
    return JsonResponse({"status": "ok", "messages": data, "title": session.title})


@login_required
@premium_required
def chat_session_send(request, session_id):
    """POST — Send a message in a session, get AI response."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)

    try:
        data = json.loads(request.body)
        message = data.get("message", "").strip()
        scan_id = data.get("scan_id", None)

        if not message:
            return JsonResponse({"error": "Message cannot be empty"}, status=400)

        if len(message) > 2000:
            return JsonResponse({"error": "Message too long (max 2000 chars)"}, status=400)

        # Save user message
        ChatMessage.objects.create(
            session=session,
            role="user",
            content=message,
        )

        # Build scan context if provided
        scan_context = None
        if scan_id:
            try:
                scan = ScanResult.objects.get(photo_id=scan_id, user=request.user)
                scan_context = {
                    "disease": scan.prediction,
                    "confidence": str(scan.confidence),
                    "ratio": str(scan.disease_ratio),
                    "stage": scan.disease_stage,
                    "yield_loss": str(scan.yield_loss) if scan.yield_loss else "N/A",
                    "recommendations": scan.ai_treatment or "",
                }
            except ScanResult.DoesNotExist:
                pass

        # Build chat context from recent messages in this session
        recent_msgs = session.messages.order_by("-created_at")[:10]
        chat_context = [
            {"role": m.role, "content": m.content}
            for m in reversed(list(recent_msgs))
        ]

        # Detect language from message or Django locale
        language = data.get("language") or getattr(request, 'LANGUAGE_CODE', 'en')

        # Get AI response via Hybrid AI Service
        from .services.hybrid_ai_service import get_hybrid_ai_service
        service = get_hybrid_ai_service()
        result = service.chat(
            message=message,
            scan_context=scan_context,
            chat_context=chat_context,
            language=language,
        )

        ai_text = result.get("text", "Sorry, I could not process your request.")
        is_offline = result.get("offline", False)

        # Save AI response
        ChatMessage.objects.create(
            session=session,
            role="ai",
            content=ai_text,
            is_offline=is_offline,
        )

        # Auto-title the session from first user message
        if session.title == "New Chat":
            title = message[:60] + ("..." if len(message) > 60 else "")
            session.title = title

        # ── Track message count & last_message_at ────────────────────
        from django.utils import timezone as tz
        session.message_count = session.messages.count()
        session.last_message_at = tz.now()
        session.save()  # Triggers updated_at

        return JsonResponse({
            "status": "ok",
            "response": ai_text,
            "offline": is_offline,
            "session_title": session.title,
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Server error: {str(e)}"}, status=500)


@login_required
@premium_required
@require_POST
def chat_session_delete(request, session_id):
    """POST — Delete a chat session."""
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    session.delete()
    return JsonResponse({"status": "ok"})


@login_required
@premium_required
@require_POST
def chat_session_close(request, session_id):
    """POST — Close/archive a chat session without deleting it."""
    session = get_object_or_404(ChatSession, pk=session_id, user=request.user)
    session.is_active = False
    session.is_closed = True
    session.save(update_fields=["is_active", "is_closed"])
    return JsonResponse({"status": "ok", "message": "Session closed successfully."})