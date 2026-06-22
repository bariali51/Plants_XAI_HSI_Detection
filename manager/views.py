# ============================================================================
# manager/views.py
# Admin Dashboard Views (Thin Layer)
# ============================================================================

import json

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from analysis.models import CustomUser, ScanResult

from .decorators import staff_required
from .models import AuditLog
from .selectors import (
    get_audit_logs_page,
    get_dashboard_stats,
    get_disease_distribution,
    get_distinct_stages,
    get_paid_subscribers,
    get_plan_distribution,
    get_payment_stats,
    get_recent_payments,
    get_recent_critical_scans,
    get_revenue_stats,
    get_scan_volume_trend,
    get_scans_page,
    get_severity_distribution,
    get_subscription_stats,
    get_top_users_by_scans,
    get_usage_stats,
    get_user_detail,
    get_user_scans,
    get_users_page,
)
from .services import (
    change_user_plan,
    delete_scan,
    delete_user,
    log_action,
    toggle_user_active,
    toggle_user_staff,
)


# ============================================================================
# Admin Dashboard
# ============================================================================

@staff_required
def admin_dashboard(request):
    """Main admin dashboard with KPIs, charts, subscription analytics."""
    stats = get_dashboard_stats()
    critical_scans = get_recent_critical_scans(limit=5)
    disease_dist = get_disease_distribution()
    severity_dist = get_severity_distribution()
    scan_trend = get_scan_volume_trend(days=30)

    # Subscription analytics
    sub_stats = get_subscription_stats()
    revenue_stats = get_revenue_stats()
    usage_stats = get_usage_stats()
    top_users = get_top_users_by_scans(limit=10)
    plan_dist = get_plan_distribution()
    payment_stats = get_payment_stats()
    recent_payments = get_recent_payments(limit=10)
    paid_subscribers = get_paid_subscribers()

    # Prepare chart data as JSON for Chart.js
    chart_data = {
        "disease_labels": [d["prediction"] for d in disease_dist],
        "disease_counts": [d["count"] for d in disease_dist],
        "severity_labels": [s["disease_stage"] for s in severity_dist],
        "severity_counts": [s["count"] for s in severity_dist],
        "trend_labels": [str(t["day"]) for t in scan_trend],
        "trend_counts": [t["count"] for t in scan_trend],
        "plan_labels": [p["plan_type"].capitalize() for p in plan_dist],
        "plan_counts": [p["count"] for p in plan_dist],
    }

    context = {
        "stats": stats,
        "critical_scans": critical_scans,
        "chart_data": json.dumps(chart_data),
        "sub_stats": sub_stats,
        "revenue_stats": revenue_stats,
        "usage_stats": usage_stats,
        "top_users": top_users,
        "payment_stats": payment_stats,
        "recent_payments": recent_payments,
        "paid_subscribers": paid_subscribers,
    }

    return render(request, "manager/dashboard.html", context)


# ============================================================================
# User Management
# ============================================================================

@staff_required
def admin_users(request):
    """Paginated user list with search and filters."""
    page = request.GET.get("page", 1)
    search = request.GET.get("q", "")
    purpose = request.GET.get("purpose", "")
    plan = request.GET.get("plan", "")
    role = request.GET.get("role", "")

    users_page = get_users_page(
        page=page,
        per_page=20,
        search=search,
        purpose_filter=purpose,
        plan_filter=plan,
        role_filter=role,
    )

    context = {
        "users_page": users_page,
        "search": search,
        "purpose_filter": purpose,
        "plan_filter": plan,
        "role_filter": role,
        "purpose_choices": CustomUser.PURPOSE_CHOICES,
    }

    return render(request, "manager/users.html", context)


@staff_required
def admin_user_detail(request, user_id):
    """Detailed view of a single user."""
    user_obj = get_user_detail(user_id)
    recent_scans = get_user_scans(user_id, limit=10)

    log_action(
        user=request.user,
        action="view_user",
        target_type="CustomUser",
        target_id=user_id,
        request=request,
    )

    context = {
        "profile": user_obj,
        "recent_scans": recent_scans,
    }

    return render(request, "manager/user_detail.html", context)


@staff_required
def admin_toggle_staff(request, user_id):
    """POST-only: Toggle staff status for a user."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    success, message = toggle_user_staff(request.user, user_id, request=request)

    if success:
        messages.success(request, message)
    else:
        messages.error(request, message)

    return redirect("manager:user_detail", user_id=user_id)


@staff_required
def admin_toggle_active(request, user_id):
    """POST-only: Toggle active status for a user."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    success, message = toggle_user_active(request.user, user_id, request=request)

    if success:
        messages.success(request, message)
    else:
        messages.error(request, message)

    return redirect("manager:user_detail", user_id=user_id)


@staff_required
def admin_delete_user(request, user_id):
    """POST-only: Delete a user account permanently."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    success, message = delete_user(request.user, user_id, request=request)

    if success:
        messages.success(request, message)
        return redirect("manager:users")
    else:
        messages.error(request, message)
        return redirect("manager:user_detail", user_id=user_id)


# ============================================================================
# Scan Management
# ============================================================================

@staff_required
def admin_scans(request):
    """Paginated scan list with search and filters."""
    page = request.GET.get("page", 1)
    search = request.GET.get("q", "")
    stage = request.GET.get("stage", "")
    sort = request.GET.get("sort", "-created_at")

    # Whitelist allowed sort fields
    allowed_sorts = [
        "created_at", "-created_at",
        "confidence", "-confidence",
        "disease_ratio", "-disease_ratio",
    ]
    if sort not in allowed_sorts:
        sort = "-created_at"

    scans_page = get_scans_page(
        page=page,
        per_page=20,
        search=search,
        stage_filter=stage,
        sort=sort,
    )

    stages = get_distinct_stages()

    context = {
        "scans_page": scans_page,
        "search": search,
        "stage_filter": stage,
        "current_sort": sort,
        "stages": stages,
    }

    return render(request, "manager/scans.html", context)


@staff_required
def admin_scan_detail(request, photo_id):
    """Detailed view of a single scan."""
    scan = get_object_or_404(
        ScanResult.objects.select_related("user").prefetch_related("progress_scans"),
        photo_id=photo_id,
    )

    followups = scan.progress_scans.all().order_by("-created_at")

    context = {
        "scan": scan,
        "followups": followups,
    }

    return render(request, "manager/scan_admin_detail.html", context)


@staff_required
def admin_delete_scan(request, photo_id):
    """POST-only: Delete a scan."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    success, message = delete_scan(request.user, photo_id, request=request)

    if success:
        messages.success(request, message)
    else:
        messages.error(request, message)

    return redirect("manager:scans")


# ============================================================================
# Audit Log
# ============================================================================

@staff_required
def admin_audit_log(request):
    """Paginated audit log viewer."""
    page = request.GET.get("page", 1)
    action_filter = request.GET.get("action", "")
    search_query = request.GET.get("q", "")

    logs_page = get_audit_logs_page(
        page=page,
        per_page=30,
        action_filter=action_filter,
        search_query=search_query,
    )

    context = {
        "logs_page": logs_page,
        "action_filter": action_filter,
        "search_query": search_query,
        "action_choices": AuditLog.ACTION_CHOICES,
    }

    return render(request, "manager/audit_log.html", context)


# ============================================================================
# Subscription Management
# ============================================================================

@staff_required
def admin_manage_subscription(request, user_id):
    """POST-only: Change a user's subscription plan."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    new_plan = request.POST.get("plan", "free")
    duration = int(request.POST.get("duration", 30))

    success, message = change_user_plan(
        admin_user=request.user,
        target_user_id=user_id,
        new_plan=new_plan,
        duration_days=duration,
        request=request,
    )

    if success:
        messages.success(request, message)
    else:
        messages.error(request, message)

    return redirect("manager:user_detail", user_id=user_id)


@staff_required
def admin_subscription_stats(request):
    """GET /manager/api/subscription-stats/ — JSON stats for AJAX refresh."""
    sub_stats = get_subscription_stats()
    revenue_stats = get_revenue_stats()
    usage_stats = get_usage_stats()

    return JsonResponse({
        "status": "ok",
        "subscription": sub_stats,
        "revenue": revenue_stats,
        "usage": usage_stats,
    })


@staff_required
def admin_revenue_overview(request):
    """Full-page Revenue Overview: paid subscribers with all details."""
    search = request.GET.get("q", "")
    plan_filter = request.GET.get("plan", "")

    subscribers = get_paid_subscribers()

    if search:
        from django.db.models import Q
        subscribers = subscribers.filter(
            Q(username__icontains=search) | Q(email__icontains=search)
        )
    if plan_filter:
        subscribers = subscribers.filter(plan_type=plan_filter)

    revenue_stats = get_revenue_stats()
    sub_stats = get_subscription_stats()

    context = {
        "subscribers": subscribers,
        "revenue_stats": revenue_stats,
        "sub_stats": sub_stats,
        "search": search,
        "plan_filter": plan_filter,
        "total_count": subscribers.count(),
    }
    return render(request, "manager/revenue_overview.html", context)


# ============================================================================
# Complaint Management (Admin)
# ============================================================================

@staff_required
def admin_complaints(request):
    """Admin complaint list with filters."""
    from analysis.models import Complaint

    status_filter = request.GET.get("status", "")
    qs = Complaint.objects.select_related("user").order_by("-created_at")

    if status_filter in ("seen", "unseen"):
        qs = qs.filter(status=status_filter)

    total = Complaint.objects.count()
    unseen = Complaint.objects.filter(status="unseen").count()

    context = {
        "complaints": qs,
        "status_filter": status_filter,
        "total_complaints": total,
        "unseen_complaints": unseen,
    }
    return render(request, "manager/complaints.html", context)


@staff_required
def admin_complaint_detail(request, complaint_id):
    """View a complaint detail + auto-mark as seen."""
    from analysis.models import Complaint

    complaint = get_object_or_404(Complaint, pk=complaint_id)

    # Auto-mark as seen when admin opens it
    if complaint.status == "unseen":
        complaint.status = "seen"
        complaint.save(update_fields=["status"])

    return render(request, "manager/complaint_detail.html", {"complaint": complaint})


@staff_required
@require_POST
def admin_mark_complaint_seen(request, complaint_id):
    """Manually mark a complaint as seen."""
    from analysis.models import Complaint

    complaint = get_object_or_404(Complaint, pk=complaint_id)
    complaint.status = "seen"
    complaint.save(update_fields=["status"])
    messages.success(request, f"Complaint #{complaint_id} marked as seen.")
    return redirect("manager:complaints")


# ============================================================================
# Expert Management
# ============================================================================

@staff_required
def admin_experts_list(request):
    """List all expert accounts."""
    from experts.models import ExpertProfile

    experts = ExpertProfile.objects.select_related("user").order_by("-created_at")

    expert_data = []
    for profile in experts:
        expert_data.append({
            "profile": profile,
            "user": profile.user,
            "user_count": profile.assigned_users_count,
            "can_accept": profile.can_accept_users,
        })

    context = {"experts": expert_data}
    return render(request, "manager/experts.html", context)


@staff_required
def admin_create_expert(request):
    """Create a new expert account."""
    errors = {}

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        specialization = request.POST.get("specialization", "").strip()
        bio = request.POST.get("bio", "").strip()

        # Validation
        if not username:
            errors["username"] = "Username is required."
        elif len(username) < 3:
            errors["username"] = "Username must be at least 3 characters."
        if not email:
            errors["email"] = "Email is required."
        if not password:
            errors["password"] = "Password is required."
        elif len(password) < 6:
            errors["password"] = "Password must be at least 6 characters."
        if not specialization:
            errors["specialization"] = "Specialization is required."

        if not errors:
            from experts.services import create_expert_account
            from manager.models import AuditLog

            success, msg, user = create_expert_account(
                admin_user=request.user,
                username=username,
                email=email,
                password=password,
                specialization=specialization,
                first_name=first_name,
                last_name=last_name,
                bio=bio,
            )

            if success:
                AuditLog.objects.create(
                    user=request.user,
                    action="create_expert",
                    target_type="user",
                    target_id=str(user.id),
                    detail=f"Created expert account: {username} ({specialization})",
                )
                messages.success(request, msg)
                return redirect("manager:experts_list")
            else:
                errors["general"] = msg

    return render(request, "manager/create_expert.html", {"errors": errors})


@staff_required
def admin_expert_detail(request, user_id):
    """View/edit expert profile and manage assigned users."""
    from analysis.models import CustomUser
    from experts.models import ExpertProfile, UserExpertRelation

    expert_user = get_object_or_404(CustomUser, pk=user_id, role="expert")

    try:
        profile = expert_user.expert_profile
    except ExpertProfile.DoesNotExist:
        messages.error(request, "Expert profile not found.")
        return redirect("manager:experts_list")

    # Get assigned users
    relations = (
        UserExpertRelation.objects
        .filter(expert=expert_user)
        .select_related("user")
    )

    # Get all regular users for assignment dropdown
    available_users = (
        CustomUser.objects
        .filter(role="user", is_active=True)
        .exclude(expert_relations__expert=expert_user)
        .order_by("username")
    )

    context = {
        "expert_user": expert_user,
        "profile": profile,
        "relations": relations,
        "available_users": available_users,
        "user_count": relations.count(),
        "max_users": 10,
    }
    return render(request, "manager/expert_detail.html", context)


@staff_required
@require_POST
def admin_assign_expert(request, user_id):
    """Assign a user to an expert."""
    from analysis.models import CustomUser
    from experts.services import assign_expert_to_user
    from manager.models import AuditLog

    expert_user = get_object_or_404(CustomUser, pk=user_id, role="expert")
    target_user_id = request.POST.get("user_id")

    if not target_user_id:
        messages.error(request, "Please select a user.")
        return redirect("manager:expert_detail", user_id=user_id)

    try:
        target_user = CustomUser.objects.get(pk=target_user_id)
    except CustomUser.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect("manager:expert_detail", user_id=user_id)

    success, msg = assign_expert_to_user(request.user, target_user, expert_user)

    if success:
        AuditLog.objects.create(
            user=request.user,
            action="assign_expert",
            target_type="user",
            target_id=str(target_user.id),
            detail=f"Assigned user '{target_user.username}' to expert '{expert_user.username}'",
        )
        messages.success(request, msg)
    else:
        messages.error(request, msg)

    return redirect("manager:expert_detail", user_id=user_id)


@staff_required
@require_POST
def admin_remove_expert_user(request, user_id, relation_id):
    """Remove a user from an expert."""
    from analysis.models import CustomUser
    from experts.models import UserExpertRelation
    from manager.models import AuditLog

    expert_user = get_object_or_404(CustomUser, pk=user_id, role="expert")
    relation = get_object_or_404(UserExpertRelation, pk=relation_id, expert=expert_user)

    target_username = relation.user.username
    relation.delete()

    AuditLog.objects.create(
        user=request.user,
        action="remove_expert_user",
        target_type="user",
        target_id=str(expert_user.id),
        detail=f"Removed user '{target_username}' from expert '{expert_user.username}'",
    )

    messages.success(request, f"User '{target_username}' removed from expert '{expert_user.username}'.")
    return redirect("manager:expert_detail", user_id=user_id)


@staff_required
def admin_toggle_expert_active(request, user_id):
    """Toggle expert active status."""
    from analysis.models import CustomUser
    from experts.models import ExpertProfile
    from manager.models import AuditLog

    expert_user = get_object_or_404(CustomUser, pk=user_id, role="expert")
    profile = get_object_or_404(ExpertProfile, user=expert_user)

    profile.is_active = not profile.is_active
    profile.save(update_fields=["is_active"])

    status = "activated" if profile.is_active else "deactivated"
    AuditLog.objects.create(
        user=request.user,
        action="toggle_expert",
        target_type="user",
        target_id=str(expert_user.id),
        detail=f"Expert '{expert_user.username}' {status}",
    )

    messages.success(request, f"Expert '{expert_user.username}' has been {status}.")
    return redirect("manager:expert_detail", user_id=user_id)


# ============================================================================
# Expert Complaints (Admin View — Separate from User Complaints)
# ============================================================================

@staff_required
def admin_expert_complaints(request):
    """View all expert complaints."""
    from experts.models import ExpertComplaint

    status_filter = request.GET.get("status", "all")

    complaints = ExpertComplaint.objects.select_related("expert").order_by("-created_at")
    if status_filter == "unseen":
        complaints = complaints.filter(status="unseen")
    elif status_filter == "seen":
        complaints = complaints.filter(status="seen")

    unseen = ExpertComplaint.objects.filter(status="unseen").count()

    context = {
        "complaints": complaints,
        "status_filter": status_filter,
        "unseen_count": unseen,
        "is_expert_complaints": True,
    }
    return render(request, "manager/expert_complaints.html", context)


@staff_required
def admin_expert_complaint_detail(request, complaint_id):
    """View an expert complaint detail."""
    from experts.models import ExpertComplaint

    complaint = get_object_or_404(ExpertComplaint, pk=complaint_id)

    if complaint.status == "unseen":
        complaint.status = "seen"
        complaint.save(update_fields=["status"])

    return render(request, "manager/expert_complaint_detail.html", {"complaint": complaint})


@staff_required
@require_POST
def admin_mark_expert_complaint_seen(request, complaint_id):
    """Mark an expert complaint as seen."""
    from experts.models import ExpertComplaint

    complaint = get_object_or_404(ExpertComplaint, pk=complaint_id)
    complaint.status = "seen"
    complaint.save(update_fields=["status"])
    messages.success(request, f"Expert complaint #{complaint_id} marked as seen.")
    return redirect("manager:expert_complaints")


# ============================================================================
# Admin Scan Upload (#2)
# ============================================================================

@staff_required
def admin_upload_scan(request):
    """Allow admin to upload and diagnose plant images — same as user flow."""
    if request.method == "POST":
        image_file = request.FILES.get("image")
        if not image_file:
            messages.error(request, "Please select an image to upload.")
            return redirect("manager:upload_scan")

        import os
        import uuid
        from django.core.files.storage import FileSystemStorage
        from django.conf import settings

        try:
            from analysis.ml_utils import (
                classifier_instance,
                estimate_disease_progress,
                ai_gemini_doctor,
            )

            # Save uploaded image
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "scans"))
            photo_id = str(uuid.uuid4())[:12]
            ext = os.path.splitext(image_file.name)[1] or ".jpg"
            filename = fs.save(f"{photo_id}_original{ext}", image_file)
            image_path = os.path.join(settings.MEDIA_ROOT, "scans", filename)

            # Run prediction
            prediction, confidence = classifier_instance.predict(image_path)

            # Estimate disease progress (GradCAM)
            progress = estimate_disease_progress(image_path)
            disease_ratio = progress.get("ratio", 0)
            disease_stage = progress.get("stage", "Unknown")
            gradcam_path = progress.get("gradcam_path", "")
            yield_loss = progress.get("yield_loss", 0)

            # Relative paths for storage
            original_rel = f"scans/{filename}"
            gradcam_rel = ""
            if gradcam_path and os.path.exists(gradcam_path):
                gradcam_rel = os.path.relpath(gradcam_path, settings.MEDIA_ROOT).replace("\\", "/")

            # Try AI doctor report
            ai_data = {}
            try:
                ai_data = ai_gemini_doctor(prediction, confidence, disease_ratio, disease_stage)
            except Exception:
                pass

            # Save scan result
            scan = ScanResult.objects.create(
                user=request.user,
                photo_id=photo_id,
                image_original=original_rel,
                image_gradcam=gradcam_rel,
                prediction=prediction,
                confidence=confidence,
                disease_ratio=disease_ratio,
                disease_stage=disease_stage,
                yield_loss=yield_loss,
                ai_medical=ai_data.get("medical", ""),
                ai_treatment=ai_data.get("treatment", ""),
                ai_irrigation=ai_data.get("irrigation", ""),
                ai_economic=ai_data.get("economic", ""),
            )

            AuditLog.objects.create(
                admin=request.user,
                action="admin_scan_upload",
                details=f"Admin uploaded scan: {prediction} ({confidence}%)",
            )

            messages.success(request, f"Scan uploaded successfully! Detected: {prediction}")
            return redirect("scan_detail", photo_id=photo_id)

        except Exception as e:
            messages.error(request, f"Error processing scan: {str(e)}")
            return redirect("manager:upload_scan")

    return render(request, "manager/admin_upload.html")


# ============================================================================
# Admin Delete Expert (#15)
# ============================================================================

@staff_required
@require_POST
def admin_delete_expert(request, user_id):
    """Delete an expert account and all related data."""
    from experts.models import ExpertProfile, UserExpertRelation, Conversation

    expert_user = get_object_or_404(CustomUser, pk=user_id)

    if not expert_user.is_expert:
        messages.error(request, f"User '{expert_user.username}' is not an expert.")
        return redirect("manager:experts_list")

    if expert_user == request.user:
        messages.error(request, "You cannot delete your own account.")
        return redirect("manager:experts_list")

    username = expert_user.username

    # Log BEFORE deletion so audit trail is preserved
    AuditLog.objects.create(
        user=request.user,
        action="delete_expert",
        target_type="CustomUser",
        target_id=str(user_id),
        detail=f"Permanently deleted expert account: {username}",
    )

    # Remove all relationships
    UserExpertRelation.objects.filter(expert=expert_user).delete()

    # Remove all conversations (both as user and as expert)
    Conversation.objects.filter(expert=expert_user).delete()
    Conversation.objects.filter(user=expert_user).delete()

    # Delete expert profile
    ExpertProfile.objects.filter(user=expert_user).delete()

    # Delete the user account (CASCADE handles remaining FKs)
    expert_user.delete()

    messages.success(request, f"Expert '{username}' has been permanently deleted.")
    return redirect("manager:experts_list")


# ============================================================================
# Admin Edit Expert Profile (#profile)
# ============================================================================

ALLOWED_AVATAR_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2 MB


@staff_required
@require_POST
def admin_edit_expert_profile(request, user_id):
    """Admin-side: update an expert's full profile."""
    import os
    from experts.models import ExpertProfile

    expert_user = get_object_or_404(CustomUser, pk=user_id, role="expert")
    profile = get_object_or_404(ExpertProfile, user=expert_user)

    first_name = request.POST.get("first_name", "").strip()
    last_name = request.POST.get("last_name", "").strip()
    title = request.POST.get("title", "").strip()
    specialization = request.POST.get("specialization", "").strip()
    bio = request.POST.get("bio", "").strip()
    skills_raw = request.POST.get("skills", "").strip()
    linkedin_url = request.POST.get("linkedin_url", "").strip()
    website_url = request.POST.get("website_url", "").strip()
    avatar_file = request.FILES.get("avatar")

    if not specialization:
        messages.error(request, "Specialization is required.")
        return redirect("manager:expert_detail", user_id=user_id)

    # Update user name
    expert_user.first_name = first_name
    expert_user.last_name = last_name
    expert_user.save(update_fields=["first_name", "last_name"])

    # Update profile fields
    profile.title = title
    profile.specialization = specialization
    profile.bio = bio
    profile.linkedin_url = linkedin_url
    profile.website_url = website_url

    # Parse skills
    skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []
    profile.skills = skills_list

    update_fields = [
        "title", "specialization", "bio",
        "skills", "linkedin_url", "website_url",
    ]

    # Handle avatar upload
    if avatar_file:
        ext = os.path.splitext(avatar_file.name)[1].lower()
        if ext not in ALLOWED_AVATAR_EXTS:
            messages.error(request, f"Avatar type '{ext}' not allowed. Use JPG, PNG, GIF, or WebP.")
            return redirect("manager:expert_detail", user_id=user_id)
        if avatar_file.size > MAX_AVATAR_SIZE:
            messages.error(request, "Avatar too large (max 2 MB).")
            return redirect("manager:expert_detail", user_id=user_id)

        if profile.avatar:
            profile.avatar.delete(save=False)
        profile.avatar = avatar_file
        update_fields.append("avatar")

    profile.save(update_fields=update_fields)

    # Audit log
    AuditLog.objects.create(
        user=request.user,
        action="edit_expert",
        target_type="ExpertProfile",
        target_id=str(user_id),
        detail=f"Updated expert profile for '{expert_user.username}'",
    )

    messages.success(request, f"Expert profile for '{expert_user.username}' updated successfully.")
    return redirect("manager:expert_detail", user_id=user_id)


# ============================================================================
# Admin ↔ Expert Messaging
# ============================================================================

@staff_required
def admin_messages_inbox(request):
    """Admin messaging inbox — list all conversations with experts."""
    from experts.models import AdminConversation, ExpertProfile

    conversations = (
        AdminConversation.objects
        .filter(admin=request.user)
        .select_related("expert")
        .order_by("-updated_at")
    )

    conv_data = []
    for conv in conversations:
        try:
            profile = conv.expert.expert_profile
        except ExpertProfile.DoesNotExist:
            profile = None

        conv_data.append({
            "conversation": conv,
            "unread": conv.unread_count_for(request.user),
            "last_msg": conv.last_message,
            "profile": profile,
        })

    # Get all experts for starting new conversations
    experts = ExpertProfile.objects.filter(is_active=True).select_related("user")

    total_unread = sum(c["unread"] for c in conv_data)

    context = {
        "conversations": conv_data,
        "experts": experts,
        "total_unread": total_unread,
    }
    return render(request, "manager/admin_messages.html", context)


@staff_required
def admin_conversation_view(request, conversation_id):
    """View a specific admin-expert conversation."""
    from experts.models import AdminConversation, AdminMessage

    conversation = get_object_or_404(
        AdminConversation, pk=conversation_id, admin=request.user
    )

    # Mark messages as read
    AdminMessage.objects.filter(
        conversation=conversation, is_read=False
    ).exclude(sender=request.user).update(is_read=True)

    messages_qs = conversation.admin_messages.select_related("sender").order_by("created_at")

    context = {
        "conversation": conversation,
        "messages_list": messages_qs,
        "other_user": conversation.expert,
    }
    return render(request, "manager/admin_conversation.html", context)


@staff_required
def admin_start_conversation(request, expert_id):
    """Start or resume a conversation with an expert."""
    from experts.models import AdminConversation

    expert_user = get_object_or_404(CustomUser, pk=expert_id, role="expert")

    conv, created = AdminConversation.objects.get_or_create(
        admin=request.user,
        expert=expert_user,
        defaults={"subject": f"Chat with {expert_user.username}"},
    )

    return redirect("manager:admin_conversation", conversation_id=conv.pk)


@staff_required
def api_admin_send_message(request):
    """POST — Send a message (text and/or image) in an admin-expert conversation."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    from experts.models import AdminConversation, AdminMessage

    # Support both FormData (with files) and JSON
    if request.content_type and "multipart" in request.content_type:
        conversation_id = request.POST.get("conversation_id")
        text = request.POST.get("text", "").strip()
        image = request.FILES.get("image")
    else:
        try:
            data = json.loads(request.body)
            conversation_id = data.get("conversation_id")
            text = data.get("text", "").strip()
            image = None
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not conversation_id or (not text and not image):
        return JsonResponse({"error": "Conversation ID and text or image required"}, status=400)

    conversation = get_object_or_404(
        AdminConversation, pk=conversation_id, admin=request.user
    )

    msg = AdminMessage.objects.create(
        conversation=conversation,
        sender=request.user,
        text=text,
        image=image,
    )

    conversation.save()  # Update updated_at

    return JsonResponse({
        "status": "ok",
        "message": {
            "id": msg.id,
            "text": msg.text,
            "image_url": msg.image.url if msg.image else None,
            "sender": msg.sender.username,
            "is_mine": True,
            "created_at": msg.created_at.strftime("%H:%M"),
            "created_at_full": msg.created_at.strftime("%Y-%m-%d %H:%M"),
        },
    })


@staff_required
def api_admin_get_messages(request, conversation_id):
    """GET — Poll for new messages in an admin-expert conversation."""
    from experts.models import AdminConversation, AdminMessage

    conversation = get_object_or_404(
        AdminConversation, pk=conversation_id, admin=request.user
    )

    after_id = request.GET.get("after", 0)
    try:
        after_id = int(after_id)
    except (ValueError, TypeError):
        after_id = 0

    new_messages = conversation.admin_messages.filter(
        id__gt=after_id
    ).select_related("sender")

    # Mark as read
    AdminMessage.objects.filter(
        conversation=conversation, is_read=False
    ).exclude(sender=request.user).update(is_read=True)

    messages_data = []
    user_id = str(request.user.id)
    for msg in new_messages:
        messages_data.append({
            "id": msg.id,
            "text": msg.text,
            "image_url": msg.image.url if msg.image else None,
            "sender": msg.sender.username,
            "is_mine": msg.sender == request.user,
            "is_deleted": msg.is_deleted,
            "reactions": {k: len(v) for k, v in (msg.reactions or {}).items()},
            "my_reactions": [k for k, v in (msg.reactions or {}).items() if user_id in v],
            "created_at": msg.created_at.strftime("%H:%M"),
            "created_at_full": msg.created_at.strftime("%Y-%m-%d %H:%M"),
        })

    return JsonResponse({"status": "ok", "messages": messages_data})


@staff_required
@require_POST
def api_admin_react_message(request, message_id):
    """POST — Toggle an emoji reaction on a message."""
    from experts.models import AdminMessage

    msg = get_object_or_404(AdminMessage, pk=message_id)
    # Verify admin is part of this conversation
    if msg.conversation.admin != request.user:
        return JsonResponse({"error": "Forbidden"}, status=403)

    try:
        data = json.loads(request.body)
        emoji = data.get("emoji", "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not emoji:
        return JsonResponse({"error": "Emoji required"}, status=400)

    reactions = msg.reactions or {}
    user_id = str(request.user.id)

    if emoji not in reactions:
        reactions[emoji] = []

    if user_id in reactions[emoji]:
        reactions[emoji].remove(user_id)
        action = "removed"
        if not reactions[emoji]:
            del reactions[emoji]
    else:
        reactions[emoji].append(user_id)
        action = "added"

    msg.reactions = reactions
    msg.save(update_fields=["reactions"])

    return JsonResponse({
        "status": "ok",
        "action": action,
        "reactions": {k: len(v) for k, v in msg.reactions.items()},
        "my_reactions": [k for k, v in msg.reactions.items() if user_id in v],
    })


@staff_required
@require_POST
def api_admin_delete_message(request, message_id):
    """POST — Soft-delete a message (only sender can delete)."""
    from experts.models import AdminMessage

    msg = get_object_or_404(AdminMessage, pk=message_id)

    if msg.sender != request.user:
        return JsonResponse({"error": "You can only delete your own messages"}, status=403)

    msg.is_deleted = True
    msg.text = ""
    if msg.image:
        msg.image.delete(save=False)
        msg.image = None
    msg.save(update_fields=["is_deleted", "text", "image"])

    return JsonResponse({"status": "ok", "message_id": message_id})
