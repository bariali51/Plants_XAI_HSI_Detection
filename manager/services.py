# ============================================================================
# manager/services.py
# Admin Business Logic & Audit Logging
# ============================================================================

import logging

from django.shortcuts import get_object_or_404

from analysis.models import CustomUser, ScanResult

from .models import AuditLog

logger = logging.getLogger(__name__)


# ============================================================================
# Audit Logging
# ============================================================================

def log_action(user, action, target_type="", target_id="", detail="", request=None):
    """Create an audit log entry for an admin action."""
    ip_address = None
    if request:
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip_address = x_forwarded_for.split(",")[0].strip()
        else:
            ip_address = request.META.get("REMOTE_ADDR")

    AuditLog.objects.create(
        user=user,
        action=action,
        target_type=target_type,
        target_id=str(target_id),
        detail=detail,
        ip_address=ip_address,
    )

    logger.info(
        "ADMIN ACTION: %s by %s on %s(%s) — %s",
        action,
        user.username if user else "system",
        target_type,
        target_id,
        detail or "no detail",
    )


# ============================================================================
# User Management
# ============================================================================

def toggle_user_staff(admin_user, target_user_id, request=None):
    """Toggle is_staff status for a user and log the action."""
    target = get_object_or_404(CustomUser, pk=target_user_id)

    if target == admin_user:
        return False, "You cannot modify your own staff status."

    if target.is_superuser:
        return False, "Cannot modify superuser status."

    target.is_staff = not target.is_staff
    target.save(update_fields=["is_staff"])

    new_status = "staff" if target.is_staff else "non-staff"
    log_action(
        user=admin_user,
        action="toggle_staff",
        target_type="CustomUser",
        target_id=target_user_id,
        detail=f"Changed {target.username} to {new_status}",
        request=request,
    )

    return True, f"{target.username} is now {new_status}."


def toggle_user_active(admin_user, target_user_id, request=None):
    """Toggle is_active status for a user and log the action."""
    target = get_object_or_404(CustomUser, pk=target_user_id)

    if target == admin_user:
        return False, "You cannot deactivate yourself."

    if target.is_superuser:
        return False, "Cannot deactivate a superuser."

    target.is_active = not target.is_active
    target.save(update_fields=["is_active"])

    action = "activate_user" if target.is_active else "deactivate_user"
    status = "activated" if target.is_active else "deactivated"

    log_action(
        user=admin_user,
        action=action,
        target_type="CustomUser",
        target_id=target_user_id,
        detail=f"{target.username} account {status}",
        request=request,
    )

    return True, f"{target.username} has been {status}."


# ============================================================================
# Scan Management
# ============================================================================

def delete_scan(admin_user, photo_id, request=None):
    """Delete a scan and log the action."""
    scan = get_object_or_404(ScanResult, photo_id=photo_id)

    detail = f"Deleted scan '{scan.prediction}' by {scan.user.username}"

    log_action(
        user=admin_user,
        action="delete_scan",
        target_type="ScanResult",
        target_id=photo_id,
        detail=detail,
        request=request,
    )

    scan.delete()
    return True, "Scan deleted successfully."


def delete_user(admin_user, target_user_id, request=None):
    """Delete a user account and all related data safely."""
    target = get_object_or_404(CustomUser, pk=target_user_id)

    # Safety checks
    if target == admin_user:
        return False, "You cannot delete your own account."

    if target.is_superuser:
        return False, "Cannot delete a superuser account."

    if target.role == "expert":
        return False, "Use the Expert Management page to delete expert accounts."

    username = target.username
    email = target.email

    # Log before deletion (so we have the audit trail)
    log_action(
        user=admin_user,
        action="delete_user",
        target_type="CustomUser",
        target_id=target_user_id,
        detail=f"Permanently deleted user '{username}' ({email})",
        request=request,
    )

    # Django CASCADE will handle: ScanResult, ChatSession, ChatMessage,
    # Complaint, FakePayment, FollowUpScan, expert_relations, etc.
    target.delete()

    return True, f"User '{username}' has been permanently deleted."


# ============================================================================
# Subscription Management
# ============================================================================

def change_user_plan(admin_user, target_user_id, new_plan, duration_days=30, request=None):
    """Change a user's subscription plan and log the action."""
    from analysis.models import PLAN_CONFIG
    from analysis.services.subscription import upgrade_user

    target = get_object_or_404(CustomUser, pk=target_user_id)

    if new_plan not in PLAN_CONFIG:
        return False, f"Invalid plan: {new_plan}"

    old_plan = target.plan_type
    success, message = upgrade_user(target, new_plan, duration_days)

    if success:
        log_action(
            user=admin_user,
            action="change_plan",
            target_type="CustomUser",
            target_id=target_user_id,
            detail=f"Changed {target.username} from {old_plan} to {new_plan} ({duration_days} days)",
            request=request,
        )

        # Auto-assign experts when upgrading to premium
        if new_plan == "premium":
            try:
                from experts.auto_linking import auto_assign_experts_to_user
                auto_assign_experts_to_user(target)
            except Exception as e:
                logger.warning("Auto-linking failed for user %s: %s", target.username, e)

    return success, message
