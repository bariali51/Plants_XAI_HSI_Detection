# ============================================================================
# analysis/context_processors.py
# Template Context Processor — Injects subscription data into all templates
# ============================================================================

from .models import PLAN_CONFIG


def subscription_context(request):
    """
    Inject subscription-related data into every template context.

    Available in templates as:
        {{ plan_type }}           - 'free', 'basic', 'premium'
        {{ plan_label }}          - 'Free', 'Basic', 'Premium'
        {{ remaining_scans }}     - int or None (unlimited)
        {{ daily_limit }}         - int or None (unlimited)
        {{ can_chat }}            - bool
        {{ is_premium }}          - bool
        {{ subscription_end }}    - datetime or None
        {{ plan_config }}         - full config dict
        {{ all_plans }}           - all plan configs
        {{ is_expert }}           - bool
        {{ is_admin_role }}       - bool
        {{ unread_messages }}     - int (expert/user messages)
        {{ has_experts }}         - bool (user has assigned experts)
    """
    user = getattr(request, "user", None)

    if user is None or not getattr(user, "is_authenticated", False):
        return {
            "plan_type": "free",
            "plan_label": "Free",
            "remaining_scans": 3,
            "daily_limit": 3,
            "can_chat": False,
            "is_premium": False,
            "subscription_end": None,
            "plan_config": PLAN_CONFIG["free"],
            "all_plans": PLAN_CONFIG,
            "is_expert": False,
            "is_admin_role": False,
            "unread_messages": 0,
            "has_experts": False,
            "unseen_complaints": 0,
            "unseen_expert_complaints": 0,
        }

    config = user.plan_config
    eff_plan = user.effective_plan_type

    # Unseen complaint count (for notification badges)
    unseen_complaints = 0
    unseen_expert_complaints = 0
    if user.is_staff:
        from .models import Complaint
        unseen_complaints = Complaint.objects.filter(status="unseen").count()
        try:
            from experts.models import ExpertComplaint
            unseen_expert_complaints = ExpertComplaint.objects.filter(status="unseen").count()
        except Exception:
            pass

    # Expert system context
    is_expert = getattr(user, "is_expert", False)
    is_admin_role = getattr(user, "is_admin_role", False)

    unread_messages = 0
    has_experts = False
    try:
        from experts.services import get_unread_count
        unread_messages = get_unread_count(user)

        if not is_expert and not user.is_staff:
            from experts.models import UserExpertRelation
            has_experts = UserExpertRelation.objects.filter(user=user).exists()
    except Exception:
        pass

    return {
        "plan_type": eff_plan,
        "plan_label": config["label"],
        "remaining_scans": user.remaining_scans_today(),
        "daily_limit": config["daily_limit"],
        "can_chat": user.can_use_chat(),
        "is_premium": eff_plan == "premium",
        "subscription_end": user.subscription_end,
        "plan_config": config,
        "all_plans": PLAN_CONFIG,
        "unseen_complaints": unseen_complaints,
        "unseen_expert_complaints": unseen_expert_complaints,
        "is_expert": is_expert,
        "is_admin_role": is_admin_role,
        "unread_messages": unread_messages,
        "has_experts": has_experts,
    }

