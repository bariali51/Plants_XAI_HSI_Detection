# ============================================================================
# experts/auto_linking.py
# Automatic Expert-User Assignment Service
# ============================================================================

import logging
from typing import List, Tuple

from django.db.models import Count

logger = logging.getLogger(__name__)

# ── Limits ───────────────────────────────────────────────────────────────────
MAX_EXPERTS_PER_USER = 3
MAX_USERS_PER_EXPERT = 10


def auto_assign_experts_to_user(user, max_experts: int = MAX_EXPERTS_PER_USER) -> List[Tuple[str, str]]:
    """
    When a user becomes Premium, automatically assign up to `max_experts` experts.

    Uses lowest-load algorithm: experts with the fewest assigned users are prioritized.

    Args:
        user: CustomUser instance (must be premium)
        max_experts: Maximum number of experts to assign

    Returns:
        List of (expert_username, status_message) tuples
    """
    from experts.models import ExpertProfile, UserExpertRelation

    results = []

    # Count how many experts this user already has
    current_count = UserExpertRelation.objects.filter(user=user).count()
    slots_available = max_experts - current_count

    if slots_available <= 0:
        logger.info(
            "AUTO-LINK: User '%s' already has %d/%d experts, skipping.",
            user.username, current_count, max_experts,
        )
        return results

    # Get already-assigned expert IDs to exclude
    assigned_expert_ids = (
        UserExpertRelation.objects
        .filter(user=user)
        .values_list("expert_id", flat=True)
    )

    # Find active experts with capacity, ordered by lowest load
    available_experts = (
        ExpertProfile.objects
        .filter(is_active=True)
        .exclude(user_id__in=assigned_expert_ids)
        .annotate(user_count=Count("user__user_relations"))
        .filter(user_count__lt=MAX_USERS_PER_EXPERT)
        .order_by("user_count")[:slots_available]
    )

    for profile in available_experts:
        try:
            UserExpertRelation.objects.create(
                user=user,
                expert=profile.user,
            )
            results.append((profile.user.username, "assigned"))
            logger.info(
                "AUTO-LINK: Assigned expert '%s' to user '%s'",
                profile.user.username, user.username,
            )
        except Exception as e:
            results.append((profile.user.username, f"failed: {e}"))
            logger.error(
                "AUTO-LINK: Failed to assign expert '%s' to user '%s': %s",
                profile.user.username, user.username, e,
            )

    if results:
        logger.info(
            "AUTO-LINK: Assigned %d experts to user '%s'",
            len([r for r in results if r[1] == "assigned"]),
            user.username,
        )

    return results


def auto_assign_users_to_expert(expert_user, max_users: int = MAX_USERS_PER_EXPERT) -> List[Tuple[str, str]]:
    """
    When an expert account is created, automatically assign up to `max_users`
    premium users who need experts.

    Prioritizes users with the fewest assigned experts.

    Args:
        expert_user: CustomUser instance (with role='expert')
        max_users: Maximum number of users to assign

    Returns:
        List of (username, status_message) tuples
    """
    from analysis.models import CustomUser
    from experts.models import UserExpertRelation

    results = []

    # Count how many users this expert already has
    current_count = UserExpertRelation.objects.filter(expert=expert_user).count()
    slots_available = max_users - current_count

    if slots_available <= 0:
        logger.info(
            "AUTO-LINK: Expert '%s' already has %d/%d users, skipping.",
            expert_user.username, current_count, max_users,
        )
        return results

    # Get already-assigned user IDs to exclude
    assigned_user_ids = (
        UserExpertRelation.objects
        .filter(expert=expert_user)
        .values_list("user_id", flat=True)
    )

    # Find premium users who need more experts, ordered by fewest experts first
    available_users = (
        CustomUser.objects
        .filter(
            role="user",
            is_active=True,
            plan_type="premium",
        )
        .exclude(pk__in=assigned_user_ids)
        .annotate(expert_count=Count("expert_relations"))
        .filter(expert_count__lt=MAX_EXPERTS_PER_USER)
        .order_by("expert_count")[:slots_available]
    )

    for target_user in available_users:
        try:
            UserExpertRelation.objects.create(
                user=target_user,
                expert=expert_user,
            )
            results.append((target_user.username, "assigned"))
            logger.info(
                "AUTO-LINK: Assigned user '%s' to expert '%s'",
                target_user.username, expert_user.username,
            )
        except Exception as e:
            results.append((target_user.username, f"failed: {e}"))
            logger.error(
                "AUTO-LINK: Failed to assign user '%s' to expert '%s': %s",
                target_user.username, expert_user.username, e,
            )

    if results:
        logger.info(
            "AUTO-LINK: Assigned %d users to expert '%s'",
            len([r for r in results if r[1] == "assigned"]),
            expert_user.username,
        )

    return results
