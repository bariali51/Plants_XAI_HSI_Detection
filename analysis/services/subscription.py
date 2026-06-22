# ============================================================================
# analysis/services/subscription.py
# Subscription Business Logic — Centralized service for plan management
# ============================================================================

import logging
from datetime import timedelta

from django.utils import timezone

from analysis.models import PLAN_CONFIG

logger = logging.getLogger(__name__)


def get_plan_config(plan_type="free"):
    """Return the full configuration dict for a given plan type."""
    return PLAN_CONFIG.get(plan_type, PLAN_CONFIG["free"])


def get_all_plans():
    """Return all plan configs as a list of dicts (for pricing pages)."""
    plans = []
    for key, cfg in PLAN_CONFIG.items():
        plans.append({
            "key": key,
            **cfg,
        })
    return plans


def check_scan_limit(user):
    """
    Check if a user can perform a scan right now.

    Returns:
        dict: {allowed, remaining, limit, plan_type, message}
    """
    user.check_and_expire_subscription()

    allowed = user.can_scan()
    remaining = user.remaining_scans_today()
    limit = user.get_daily_limit()

    if allowed:
        message = ""
    elif limit is not None:
        message = (
            f"Daily scan limit reached ({limit}/{limit}). "
            f"Upgrade your plan for more scans."
        )
    else:
        message = "You cannot perform scans at this time."

    return {
        "allowed": allowed,
        "remaining": remaining,
        "limit": limit,
        "plan_type": user.plan_type,
        "message": message,
    }


def record_scan(user):
    """Record a successful scan for the user."""
    user.increment_scan_count()
    logger.info(
        "SCAN RECORDED: user=%s plan=%s daily=%d total=%d",
        user.username,
        user.plan_type,
        user.daily_scan_count,
        user.total_scans_count,
    )


def check_chat_access(user):
    """
    Check if a user can access the AI chat.

    Returns:
        tuple: (allowed: bool, message: str)
    """
    user.check_and_expire_subscription()

    if user.can_use_chat():
        return True, ""
    return False, "Upgrade to Premium to use AI Assistant"


def validate_subscription(user):
    """
    Validate and auto-expire a user's subscription if needed.

    Returns:
        bool: True if subscription was expired during this call
    """
    expired = user.check_and_expire_subscription()
    if expired:
        logger.info(
            "SUBSCRIPTION EXPIRED: user=%s downgraded to free",
            user.username,
        )
    return expired


def upgrade_user(user, plan_type, duration_days=30):
    """
    Upgrade a user to a new plan.

    Args:
        user: CustomUser instance
        plan_type: 'basic' or 'premium'
        duration_days: subscription duration (default 30)

    Returns:
        tuple: (success: bool, message: str)
    """
    if plan_type not in PLAN_CONFIG:
        return False, f"Invalid plan type: {plan_type}"

    now = timezone.now()
    user.plan_type = plan_type
    user.subscription_start = now

    if plan_type == "free":
        user.subscription_end = None
    else:
        user.subscription_end = now + timedelta(days=duration_days)

    user.save(update_fields=[
        "plan_type", "subscription_start", "subscription_end",
    ])

    logger.info(
        "PLAN UPGRADE: user=%s plan=%s end=%s",
        user.username,
        plan_type,
        user.subscription_end,
    )

    return True, f"User {user.username} upgraded to {plan_type}."
