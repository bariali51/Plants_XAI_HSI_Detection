# ============================================================================
# community/recommendations.py
# Agricultural Social Platform — Tag-Based Recommendation Engine
# ============================================================================

import logging
from django.db.models import Count, Q

logger = logging.getLogger(__name__)


def recommend_companies_for_post(post, limit=5):
    """
    Recommend companies whose tags overlap with the post's tags.
    Score = number of shared tags. Verified companies get a boost.
    """
    from companies.models import CompanyProfile

    post_tag_ids = list(post.tags.values_list("id", flat=True))
    if not post_tag_ids:
        return []

    companies = (
        CompanyProfile.objects
        .filter(is_active=True, tags__id__in=post_tag_ids)
        .annotate(match_score=Count("tags", filter=Q(tags__id__in=post_tag_ids)))
        .order_by("-verified", "-match_score")
        .distinct()[:limit]
    )

    return [
        {
            "id": c.pk,
            "name": c.company_name,
            "logo": c.logo.url if c.logo else None,
            "verified": c.verified,
            "rating": c.average_rating,
            "match_score": c.match_score,
            "specializations": c.specializations,
        }
        for c in companies
    ]


def recommend_experts_for_post(post, limit=3):
    """
    Recommend experts by matching their skills/specialization with post tags.
    """
    from experts.models import ExpertProfile

    post_tag_names = list(post.tags.values_list("name", flat=True))
    if not post_tag_names:
        return []

    post_tag_names_lower = {name.lower() for name in post_tag_names}

    active_experts = (
        ExpertProfile.objects
        .filter(is_active=True)
        .select_related("user")
    )

    matched_experts = []
    for e in active_experts:
        # Check specialization (case-insensitive substring match)
        spec = (e.specialization or "").lower()
        if any(tag in spec for tag in post_tag_names_lower):
            matched_experts.append(e)
            continue
        
        # Check skills (case-insensitive match in skills list)
        skills = e.skills or []
        skills_lower = [str(s).lower() for s in skills]
        if any(tag in skills_lower for tag in post_tag_names_lower):
            matched_experts.append(e)
            continue

    experts = matched_experts[:limit]

    return [
        {
            "id": e.user.pk,
            "name": e.display_name,
            "specialization": e.specialization,
            "avatar": e.avatar.url if e.avatar else None,
        }
        for e in experts
    ]


def recommend_similar_posts(post, limit=5):
    """
    Find posts with similar tags, preferring resolved posts (solved cases).
    """
    from community.models import Post

    post_tag_ids = list(post.tags.values_list("id", flat=True))
    if not post_tag_ids:
        return []

    similar = (
        Post.objects
        .filter(tags__id__in=post_tag_ids)
        .exclude(pk=post.pk)
        .annotate(match_score=Count("tags", filter=Q(tags__id__in=post_tag_ids)))
        .order_by("-match_score", "-created_at")
        .select_related("author")
        .distinct()[:limit]
    )

    return [
        {
            "id": p.pk,
            "title": p.title,
            "author": p.author.username,
            "status": p.status,
            "post_type": p.post_type,
            "match_score": p.match_score,
            "created_at": p.created_at.strftime("%b %d, %Y"),
        }
        for p in similar
    ]


def recommend_for_user(user, limit=5):
    """
    Personalized recommendations based on user's scan history and post activity.
    """
    from analysis.models import ScanResult
    from companies.models import CompanyProfile
    from community.models import Post, Tag

    # Collect disease/crop keywords from scan history
    recent_scans = ScanResult.objects.filter(user=user).order_by("-created_at")[:10]
    keywords = set()
    for scan in recent_scans:
        if scan.prediction:
            for word in scan.prediction.replace("_", " ").split():
                if len(word) > 2:
                    keywords.add(word.lower())

    # Collect tag names from user's posts
    user_tag_names = list(
        Tag.objects
        .filter(posts__author=user)
        .values_list("name", flat=True)
        .distinct()
    )
    keywords.update(t.lower() for t in user_tag_names)

    if not keywords:
        return {"companies": [], "posts": []}

    # Match companies
    q_company = Q()
    for kw in keywords:
        q_company |= Q(specializations__icontains=kw)
        q_company |= Q(tags__name__icontains=kw)

    companies = (
        CompanyProfile.objects
        .filter(is_active=True)
        .filter(q_company)
        .distinct()
        .order_by("-verified")[:limit]
    )

    # Match posts
    q_posts = Q()
    for kw in keywords:
        q_posts |= Q(title__icontains=kw) | Q(tags__name__icontains=kw)

    posts = (
        Post.objects
        .filter(q_posts)
        .exclude(author=user)
        .distinct()
        .order_by("-created_at")[:limit]
    )

    return {
        "companies": [
            {"id": c.pk, "name": c.company_name, "verified": c.verified}
            for c in companies
        ],
        "posts": [
            {"id": p.pk, "title": p.title, "status": p.status}
            for p in posts
        ],
    }
