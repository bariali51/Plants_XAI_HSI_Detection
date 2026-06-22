# ============================================================================
# manager/selectors.py
# Optimized ORM Queries for Admin Dashboard
# ============================================================================

from datetime import timedelta

from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.utils import timezone

from analysis.models import CustomUser, FollowUpScan, ScanResult


# ============================================================================
# Dashboard KPI Stats
# ============================================================================

def get_dashboard_stats():
    """Return aggregate KPI data for the admin dashboard."""
    now = timezone.now()
    seven_days_ago = now - timedelta(days=7)

    total_users = CustomUser.objects.count()
    total_scans = ScanResult.objects.count()
    scans_today = ScanResult.objects.filter(created_at__date=now.date()).count()
    critical_scans = ScanResult.objects.filter(disease_ratio__gte=70).count()
    new_users_7d = CustomUser.objects.filter(date_joined__gte=seven_days_ago).count()
    scans_7d = ScanResult.objects.filter(created_at__gte=seven_days_ago).count()
    avg_confidence = ScanResult.objects.aggregate(avg=Avg("confidence"))["avg"] or 0
    total_followups = FollowUpScan.objects.count()

    return {
        "total_users": total_users,
        "total_scans": total_scans,
        "scans_today": scans_today,
        "critical_scans": critical_scans,
        "new_users_7d": new_users_7d,
        "scans_7d": scans_7d,
        "avg_confidence": round(avg_confidence, 1),
        "total_followups": total_followups,
    }


# ============================================================================
# Chart Data
# ============================================================================

def get_disease_distribution():
    """Top 10 diseases by scan frequency (for pie/doughnut chart)."""
    return list(
        ScanResult.objects.values("prediction")
        .annotate(count=Count("id"))
        .order_by("-count")[:10]
    )


def get_scan_volume_trend(days=30):
    """Daily scan count for the last N days (for line chart)."""
    now = timezone.now()
    start = now - timedelta(days=days)

    scans = (
        ScanResult.objects.filter(created_at__gte=start)
        .extra(select={"day": "date(created_at)"})
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )

    return list(scans)


def get_severity_distribution():
    """Distribution of disease stages (for chart)."""
    return list(
        ScanResult.objects.values("disease_stage")
        .annotate(count=Count("id"))
        .order_by("-count")
    )


def get_user_registration_trend(days=30):
    """Daily user registrations for the last N days."""
    now = timezone.now()
    start = now - timedelta(days=days)

    users = (
        CustomUser.objects.filter(date_joined__gte=start)
        .extra(select={"day": "date(date_joined)"})
        .values("day")
        .annotate(count=Count("id"))
        .order_by("day")
    )

    return list(users)


# ============================================================================
# Paginated Lists
# ============================================================================

def get_scans_page(page=1, per_page=20, search="", stage_filter="", sort="-created_at"):
    """
    Paginated, filtered, searchable scan list.
    Uses select_related to avoid N+1 on user foreign key.
    """
    qs = ScanResult.objects.select_related("user").order_by(sort)

    if search:
        qs = qs.filter(
            Q(prediction__icontains=search)
            | Q(user__username__icontains=search)
            | Q(photo_id__icontains=search)
        )

    if stage_filter:
        qs = qs.filter(disease_stage=stage_filter)

    paginator = Paginator(qs, per_page)
    return paginator.get_page(page)


def get_users_page(page=1, per_page=20, search="", purpose_filter="", plan_filter="", role_filter=""):
    """
    Paginated user list annotated with scan count.
    Returns all regular accounts (farmers, companies, researchers)
    — excludes staff, superusers, and expert accounts.
    """
    qs = CustomUser.objects.filter(
        is_staff=False, is_superuser=False
    ).exclude(
        role="expert"
    ).annotate(
        scan_count=Count("scanresult")
    ).order_by("-date_joined")

    if search:
        qs = qs.filter(
            Q(username__icontains=search)
            | Q(email__icontains=search)
            | Q(user_code__icontains=search)
        )

    if purpose_filter:
        qs = qs.filter(purpose=purpose_filter)

    if plan_filter:
        qs = qs.filter(plan_type=plan_filter)

    if role_filter:
        qs = qs.filter(role=role_filter)

    paginator = Paginator(qs, per_page)
    return paginator.get_page(page)


def get_user_detail(user_id):
    """Get a user with annotated stats."""
    user = (
        CustomUser.objects.annotate(
            scan_count=Count("scanresult"),
        )
        .get(pk=user_id)
    )
    return user


def get_user_scans(user_id, limit=10):
    """Get recent scans for a specific user."""
    return ScanResult.objects.filter(user_id=user_id).order_by("-created_at")[:limit]


def get_recent_critical_scans(limit=10):
    """Get recent scans with critical severity (ratio >= 70)."""
    return (
        ScanResult.objects.select_related("user")
        .filter(disease_ratio__gte=70)
        .order_by("-created_at")[:limit]
    )


def get_audit_logs_page(page=1, per_page=30, action_filter="", search_query=""):
    """Paginated audit logs with user preloaded."""
    from .models import AuditLog
    from django.db.models import Q

    qs = AuditLog.objects.select_related("user").order_by("-created_at")

    if action_filter:
        qs = qs.filter(action=action_filter)

    if search_query:
        qs = qs.filter(
            Q(detail__icontains=search_query) |
            Q(target_type__icontains=search_query) |
            Q(target_id__icontains=search_query)
        )

    paginator = Paginator(qs, per_page)
    return paginator.get_page(page)


# ============================================================================
# Disease Stages (for filter dropdowns)
# ============================================================================

def get_distinct_stages():
    """Return distinct disease stages for filter dropdown."""
    return (
        ScanResult.objects.values_list("disease_stage", flat=True)
        .distinct()
        .order_by("disease_stage")
    )


# ============================================================================
# Subscription Analytics Selectors
# ============================================================================

def get_subscription_stats():
    """Count users per plan and active subscriptions."""
    now = timezone.now()

    free_count = CustomUser.objects.filter(plan_type="free").count()
    basic_count = CustomUser.objects.filter(plan_type="basic").count()
    premium_count = CustomUser.objects.filter(plan_type="premium").count()

    active_subscriptions = CustomUser.objects.filter(
        plan_type__in=["basic", "premium"],
        subscription_end__gt=now,
    ).count()

    return {
        "free_count": free_count,
        "basic_count": basic_count,
        "premium_count": premium_count,
        "active_subscriptions": active_subscriptions,
    }


def get_revenue_stats():
    """
    Calculate total revenue:
    - Subscriptions (active Basic + Premium users × plan price)
    - Community promotions (paid PromotedPost: status=active or expired)
    """
    from analysis.models import PLAN_CONFIG
    from community.models import PromotedPost
    from django.db.models import Sum

    now = timezone.now()

    # ── Subscription revenue ──────────────────────────────────────────
    active_basic = CustomUser.objects.filter(
        plan_type="basic",
        subscription_end__gt=now,
    ).count()

    active_premium = CustomUser.objects.filter(
        plan_type="premium",
        subscription_end__gt=now,
    ).count()

    basic_price   = PLAN_CONFIG["basic"]["price"]
    premium_price = PLAN_CONFIG["premium"]["price"]

    revenue_basic   = active_basic   * basic_price
    revenue_premium = active_premium * premium_price

    # ── Community ads revenue ─────────────────────────────────────────
    # Count promotions that were actually paid (active or completed/expired)
    promo_qs = PromotedPost.objects.filter(
        status__in=["active", "expired"]
    ).aggregate(total=Sum("amount"))
    revenue_promotions = int(promo_qs["total"] or 0)

    # Promotion counts for breakdown
    promo_active_count  = PromotedPost.objects.filter(status="active").count()
    promo_expired_count = PromotedPost.objects.filter(status="expired").count()
    promo_pending_count = PromotedPost.objects.filter(status="pending").count()
    promo_total_count   = PromotedPost.objects.count()

    total_revenue = revenue_basic + revenue_premium + revenue_promotions

    return {
        # Subscription breakdown
        "total_monthly_revenue": total_revenue,
        "revenue_basic":         revenue_basic,
        "revenue_premium":       revenue_premium,
        "active_basic":          active_basic,
        "active_premium":        active_premium,
        # Community ads breakdown
        "revenue_promotions":    revenue_promotions,
        "promo_active_count":    promo_active_count,
        "promo_expired_count":   promo_expired_count,
        "promo_pending_count":   promo_pending_count,
        "promo_total_count":     promo_total_count,
    }


def get_usage_stats():
    """Get scan usage statistics per plan."""
    now = timezone.now()

    total_scans_today = ScanResult.objects.filter(
        created_at__date=now.date()
    ).count()

    # Scans per plan (using user's current plan)
    from django.db.models import F
    scans_by_plan = list(
        ScanResult.objects.values(plan=F("user__plan_type"))
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    # Average scans per user
    total_users = CustomUser.objects.count()
    total_scans = ScanResult.objects.count()
    avg_scans = round(total_scans / max(total_users, 1), 1)

    return {
        "total_scans_today": total_scans_today,
        "scans_by_plan": scans_by_plan,
        "avg_scans_per_user": avg_scans,
    }


def get_top_users_by_scans(limit=10):
    """Get top users by total scan count."""
    return list(
        CustomUser.objects.filter(total_scans_count__gt=0)
        .order_by("-total_scans_count")[:limit]
        .values(
            "id", "username", "email", "plan_type",
            "total_scans_count", "daily_scan_count",
        )
    )


def get_plan_distribution():
    """User distribution by plan (for pie chart)."""
    return list(
        CustomUser.objects.values("plan_type")
        .annotate(count=Count("id"))
        .order_by("-count")
    )


# ============================================================================
# Fake Payment Analytics
# ============================================================================

def get_payment_stats():
    """Get fake payment statistics."""
    from analysis.models import FakePayment

    total_payments = FakePayment.objects.count()
    approved_payments = FakePayment.objects.filter(status="approved").count()

    # Upgrades per plan
    upgrades_basic = FakePayment.objects.filter(
        plan_requested="basic", status="approved"
    ).count()
    upgrades_premium = FakePayment.objects.filter(
        plan_requested="premium", status="approved"
    ).count()

    return {
        "total_payments": total_payments,
        "approved_payments": approved_payments,
        "upgrades_basic": upgrades_basic,
        "upgrades_premium": upgrades_premium,
    }


def get_recent_payments(limit=20):
    """Get recent fake payments for admin listing."""
    from analysis.models import FakePayment

    return list(
        FakePayment.objects.select_related("user")
        .order_by("-created_at")[:limit]
    )


def get_paid_subscribers():
    """
    Return all users with an active paid subscription (basic or premium),
    annotated with scan count, community post count, and community ads revenue.
    Used for the Revenue Overview page and dashboard card.
    """
    from django.db.models import Q, Sum, DecimalField
    from django.db.models.functions import Coalesce
    from django.utils import timezone as tz

    now = tz.now()
    paid_promo_filter = Q(promotions__status__in=["active", "expired"])

    return (
        CustomUser.objects.filter(
            plan_type__in=["basic", "premium"],
            subscription_end__gt=now,
            is_staff=False,
            is_superuser=False,
        )
        .exclude(role="expert")
        .annotate(
            scan_count=Count("scanresult"),
            post_count=Count("community_posts"),
            promotion_revenue=Coalesce(
                Sum("promotions__amount", filter=paid_promo_filter),
                0,
                output_field=DecimalField(),
            ),
        )
        .order_by("-subscription_end")
    )


