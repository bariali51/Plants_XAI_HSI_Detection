# ============================================================================
# community/views.py
# Agricultural Social Platform — Community Views
# ============================================================================

import hashlib
import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import models
from django.db.models import Q, Count, F, FloatField, Value
from django.db.models.functions import Coalesce
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from analysis.models import CustomUser
from .forms import PostForm, CommentForm
from .models import (
    Bookmark, Comment, CommentLike, Post, PostImage, PostLike, Tag,
    PromotedPost,
)

logger = logging.getLogger(__name__)

POSTS_PER_PAGE = 12


# ============================================================================
# Feed (Main Community Page)
# ============================================================================

@login_required
def feed_view(request):
    """
    Smart Community Feed with Paid Promotion Ranking.

    Ranking algorithm (score per post):
      1. Paid boost score  (promoted & live  → large weight)
      2. Pinned posts      (admin-pinned      → always on top)
      3. Interest match    (user crops/role  → relevance bonus)
      4. Engagement        (likes + comments  → quality signal)
      5. Recency           (recent posts      → decay penalty)

    Promoted posts that do NOT match the viewer's role/interest/location
    are silently excluded from the boost but still appear in feed.
    """
    from datetime import timedelta
    now = timezone.now()

    # ── Auto-expire stale promotions (lightweight, no celery needed) ──────
    expired_promo_post_ids = list(
        PromotedPost.objects.filter(
            status="active", end_date__lt=now
        ).values_list("post_id", flat=True)
    )
    if expired_promo_post_ids:
        PromotedPost.objects.filter(
            status="active", end_date__lt=now
        ).update(status="expired")
        # Clear is_ad flag on posts whose promotions just expired
        Post.objects.filter(pk__in=expired_promo_post_ids).update(is_ad=False)

    # ── Base queryset ──────────────────────────────────────────────────
    posts = Post.objects.select_related("author").prefetch_related(
        "tags", "images",
        "promotion",
    )

    # ── Filters from query params ──────────────────────────────────
    post_type  = request.GET.get("type")
    status_f   = request.GET.get("status")
    tag_filter = request.GET.get("tag")
    sort       = request.GET.get("sort", "smart")  # default: smart ranking

    if post_type and post_type in ("question", "solution", "discussion", "alert"):
        posts = posts.filter(post_type=post_type)
    if status_f and status_f in ("active", "resolved", "closed"):
        posts = posts.filter(status=status_f)
    if tag_filter:
        posts = posts.filter(tags__slug=tag_filter)

    posts = posts.annotate(
        likes_count=Count("likes", distinct=True),
        comments_count=Count("comments", distinct=True),
    )

    # ── Basic sort modes ────────────────────────────────────────────
    if sort == "popular":
        posts = posts.order_by("-is_pinned", "-likes_count", "-created_at")
    elif sort == "discussed":
        posts = posts.order_by("-is_pinned", "-comments_count", "-created_at")
    elif sort == "latest":
        posts = posts.order_by("-is_pinned", "-created_at")
    elif sort == "unanswered":
        posts = posts.filter(post_type="question", status="active").order_by("-created_at")
    else:
        # ── SMART RANKING (default) ──────────────────────────────
        #  Fetch all candidates, score them in Python (keeps DB simple)
        user   = request.user
        u_role = getattr(user, "role", "user")
        u_city = getattr(user, "location", "") or ""
        u_crops = getattr(user, "crops_of_interest", []) or []

        # Live promoted post IDs and their boost scores (viewer-filtered)
        live_promotions = {}
        promos = PromotedPost.objects.filter(
            status="active", end_date__gte=now
        ).select_related("post")
        for promo in promos:
            # Role match check
            if promo.target_role != "all" and promo.target_role != u_role:
                continue
            # City match check (soft — just a bonus, not a filter)
            city_bonus = 1.0 if (promo.target_city and promo.target_city.lower() in u_city.lower()) else 0.0
            live_promotions[promo.post_id] = promo.boost_score + city_bonus

        # Interest-to-tag mapping for relevance scoring
        INTEREST_KEYWORDS = {
            "pesticides":  ["pesticide", "medicine", "fungicide", "herbicide", "insecticide"],
            "crops":       ["crop", "wheat", "barley", "potato", "tomato", "corn", "cotton"],
            "equipment":   ["equipment", "tractor", "machine", "irrigation", "drone"],
            "fertilizers": ["fertilizer", "nitrogen", "phosphorus", "organic"],
            "general":     ["farm", "agriculture", "soil", "weather", "harvest"],
        }

        # Score every post
        scored = []
        for post in posts:
            score = 0.0

            # 1. Pinned always wins
            if post.is_pinned:
                score += 1000.0

            # 2. Paid promotion boost (targeting already filtered above)
            if post.pk in live_promotions:
                score += 200.0 + live_promotions[post.pk]

            # 3. Role relevance bonus
            if post.author.role == u_role:
                score += 15.0

            # 4. Interest / crop relevance bonus
            post_tag_names = [t.name.lower() for t in post.tags.all()]
            for crop in u_crops:
                if any(crop.lower() in tn for tn in post_tag_names):
                    score += 10.0
                    break

            # 5. Engagement signal (log-scaled to prevent domination)
            import math
            engagement = post.likes_count + post.comments_count
            score += math.log1p(engagement) * 5.0

            # 6. Recency decay  (-1 per hour for first 48h, then flat)
            age_hours = (now - post.created_at).total_seconds() / 3600
            if age_hours < 48:
                score += max(0, 48 - age_hours) * 0.5

            scored.append((score, post))

        # Sort descending by score, return just the posts
        scored.sort(key=lambda x: x[0], reverse=True)
        posts = [p for _, p in scored]

        # Increment impressions for live promoted posts in this page
        promoted_ids_in_feed = [
            p.pk for p in posts if p.pk in live_promotions
        ]
        if promoted_ids_in_feed:
            PromotedPost.objects.filter(
                post_id__in=promoted_ids_in_feed
            ).update(impressions=models.F("impressions") + 1)

        # Paginate the scored list directly
        paginator = Paginator(posts, POSTS_PER_PAGE)
        page = paginator.get_page(request.GET.get("page", 1))

        # Mark which posts are actively promoted (for template badge)
        promoted_post_ids = set(live_promotions.keys())

        # ── Sidebar widgets ───────────────────────────────────────────
        week_ago = now - timedelta(days=7)
        trending_posts = (
            Post.objects
            .filter(created_at__gte=week_ago)
            .select_related("author")
            .annotate(
                engagement=Count("likes", distinct=True) + Count("comments", distinct=True),
            )
            .order_by("-engagement")[:5]
        )
        popular_tags = Tag.objects.annotate(post_count=Count("posts")).order_by("-post_count")[:20]
        total_posts = Post.objects.count()
        total_resolved = Post.objects.filter(status="resolved").count()
        active_discussions = Post.objects.filter(status="active", post_type="discussion").count()

        return render(request, "community/feed.html", {
            "posts": page,
            "popular_tags": popular_tags,
            "trending_posts": trending_posts,
            "recommended_posts": Post.objects.none(),
            "promoted_post_ids": promoted_post_ids,
            "current_type": post_type,
            "current_status": status_f,
            "current_tag": tag_filter,
            "current_sort": sort,
            "total_posts": total_posts,
            "total_resolved": total_resolved,
            "active_discussions": active_discussions,
        })

    # ── Non-smart sort paths (latest / popular / discussed) ──────────
    paginator = Paginator(posts, POSTS_PER_PAGE)
    page = paginator.get_page(request.GET.get("page", 1))

    # Sidebar
    week_ago = now - timedelta(days=7)
    trending_posts = (
        Post.objects
        .filter(created_at__gte=week_ago)
        .select_related("author")
        .annotate(
            engagement=Count("likes", distinct=True) + Count("comments", distinct=True),
        )
        .order_by("-engagement")[:5]
    )

    popular_tags = Tag.objects.annotate(post_count=Count("posts")).order_by("-post_count")[:20]
    total_posts = Post.objects.count()
    total_resolved = Post.objects.filter(status="resolved").count()
    active_discussions = Post.objects.filter(status="active", post_type="discussion").count()
    promoted_post_ids = set(
        PromotedPost.objects.filter(
            status="active", end_date__gte=now
        ).values_list("post_id", flat=True)
    )

    context = {
        "posts": page,
        "popular_tags": popular_tags,
        "trending_posts": trending_posts,
        "recommended_posts": Post.objects.none(),
        "promoted_post_ids": promoted_post_ids,
        "current_type": post_type,
        "current_status": status_f,
        "current_tag": tag_filter,
        "current_sort": sort,
        "total_posts": total_posts,
        "total_resolved": total_resolved,
        "active_discussions": active_discussions,
    }
    return render(request, "community/feed.html", context)


# ============================================================================
# Post Detail
# ============================================================================

@login_required
def post_detail(request, post_id):
    """Display a single post with comments and recommendations."""
    post = get_object_or_404(
        Post.objects.select_related("author").prefetch_related("tags", "images"),
        pk=post_id,
    )

    # Increment view count
    Post.objects.filter(pk=post_id).update(view_count=post.view_count + 1)

    # Get threaded comments (top-level only, replies loaded via related)
    comments = (
        post.comments
        .filter(parent__isnull=True)
        .select_related("author")
        .prefetch_related("replies__author", "likes")
        .annotate(likes_count=Count("likes", distinct=True))
        .order_by("-is_accepted_answer", "-is_expert_response", "-is_company_response", "created_at")
    )

    # Check user interactions
    user_liked = PostLike.objects.filter(post=post, user=request.user).exists()
    user_bookmarked = Bookmark.objects.filter(post=post, user=request.user).exists()

    comment_form = CommentForm()

    context = {
        "post": post,
        "comments": comments,
        "comment_form": comment_form,
        "user_liked": user_liked,
        "user_bookmarked": user_bookmarked,
    }
    return render(request, "community/post_detail.html", context)


# ============================================================================
# Create Post
# ============================================================================

@login_required
def create_post(request):
    """Create a new community post with optional images and tags."""
    if request.method == "POST":
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()

            # Handle tags (comma-separated)
            tag_names = form.cleaned_data.get("tag_names", "")
            if tag_names:
                for name in tag_names.split(","):
                    name = name.strip().lower()
                    if name:
                        tag, _ = Tag.objects.get_or_create(
                            name=name,
                            defaults={"slug": name.replace(" ", "-")},
                        )
                        post.tags.add(tag)

            # Handle image uploads (up to 4)
            images = request.FILES.getlist("images")
            for i, img in enumerate(images[:4]):
                PostImage.objects.create(
                    post=post,
                    image=img,
                    order=i,
                )

            messages.success(request, "Post published successfully!")
            return redirect("community:post_detail", post_id=post.pk)
    else:
        form = PostForm()

    all_tags = Tag.objects.all().order_by("name")
    return render(request, "community/create_post.html", {
        "form": form,
        "all_tags": all_tags,
    })


# ============================================================================
# Edit Post
# ============================================================================

@login_required
def edit_post(request, post_id):
    """Edit an existing post (author only)."""
    post = get_object_or_404(Post, pk=post_id, author=request.user)

    if request.method == "POST":
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            form.save()

            # Update tags
            tag_names = form.cleaned_data.get("tag_names", "")
            post.tags.clear()
            if tag_names:
                for name in tag_names.split(","):
                    name = name.strip().lower()
                    if name:
                        tag, _ = Tag.objects.get_or_create(
                            name=name,
                            defaults={"slug": name.replace(" ", "-")},
                        )
                        post.tags.add(tag)

            # Handle new images
            images = request.FILES.getlist("images")
            if images:
                post.images.all().delete()
                for i, img in enumerate(images[:4]):
                    PostImage.objects.create(post=post, image=img, order=i)

            messages.success(request, "Post updated successfully!")
            return redirect("community:post_detail", post_id=post.pk)
    else:
        form = PostForm(instance=post)
        form.initial["tag_names"] = ", ".join(t.name for t in post.tags.all())

    return render(request, "community/edit_post.html", {
        "form": form,
        "post": post,
    })


# ============================================================================
# Delete Post
# ============================================================================

@login_required
@require_POST
def delete_post(request, post_id):
    """Delete a post (author only)."""
    post = get_object_or_404(Post, pk=post_id, author=request.user)
    post.delete()
    messages.success(request, "Post deleted successfully.")
    return redirect("community:feed")


# ============================================================================
# Search
# ============================================================================

@login_required
def search_posts(request):
    """Full-text search across post titles and bodies."""
    query = request.GET.get("q", "").strip()
    results = Post.objects.none()

    if query and len(query) >= 2:
        results = (
            Post.objects
            .filter(Q(title__icontains=query) | Q(body__icontains=query))
            .select_related("author")
            .prefetch_related("tags")
            .annotate(
                likes_count=Count("likes", distinct=True),
                comments_count=Count("comments", distinct=True),
            )
        )

    paginator = Paginator(results, POSTS_PER_PAGE)
    page = paginator.get_page(request.GET.get("page", 1))

    return render(request, "community/search_results.html", {
        "posts": page,
        "query": query,
        "result_count": paginator.count,
    })


# ============================================================================
# Posts by Tag
# ============================================================================

@login_required
def posts_by_tag(request, tag_slug):
    """Filter posts by a specific tag."""
    tag = get_object_or_404(Tag, slug=tag_slug)
    posts = (
        Post.objects
        .filter(tags=tag)
        .select_related("author")
        .prefetch_related("tags")
        .annotate(
            likes_count=Count("likes", distinct=True),
            comments_count=Count("comments", distinct=True),
        )
    )

    paginator = Paginator(posts, POSTS_PER_PAGE)
    page = paginator.get_page(request.GET.get("page", 1))

    return render(request, "community/feed.html", {
        "posts": page,
        "current_tag_obj": tag,
        "popular_tags": Tag.objects.annotate(post_count=Count("posts")).order_by("-post_count")[:20],
    })


# ============================================================================
# User Profile
# ============================================================================

@login_required
def user_profile(request, user_id):
    """Public community profile showing user's posts and activity."""
    profile_user = get_object_or_404(CustomUser, pk=user_id)
    user_posts = (
        Post.objects
        .filter(author=profile_user)
        .select_related("author")
        .prefetch_related("tags")
        .annotate(
            likes_count=Count("likes", distinct=True),
            comments_count=Count("comments", distinct=True),
        )
        .order_by("-created_at")[:20]
    )

    return render(request, "community/user_profile.html", {
        "profile_user": profile_user,
        "user_posts": user_posts,
        "post_count": Post.objects.filter(author=profile_user).count(),
        "comment_count": Comment.objects.filter(author=profile_user).count(),
    })


# ============================================================================
# My Bookmarks
# ============================================================================

@login_required
def my_bookmarks(request):
    """Display user's bookmarked posts."""
    bookmarks = (
        Bookmark.objects
        .filter(user=request.user)
        .select_related("post__author")
        .prefetch_related("post__tags")
        .order_by("-created_at")
    )

    return render(request, "community/feed.html", {
        "posts": [b.post for b in bookmarks],
        "page_title": "My Bookmarks",
        "is_bookmarks_page": True,
        "popular_tags": Tag.objects.annotate(post_count=Count("posts")).order_by("-post_count")[:20],
    })


# ============================================================================
# AJAX: Add Comment
# ============================================================================

@login_required
@require_POST
def api_add_comment(request):
    """AJAX endpoint to add a comment to a post."""
    try:
        data = json.loads(request.body)
        post_id = data.get("post_id")
        body = data.get("body", "").strip()
        parent_id = data.get("parent_id")

        if not body:
            return JsonResponse({"error": "Comment cannot be empty."}, status=400)
        if len(body) > 5000:
            return JsonResponse({"error": "Comment too long."}, status=400)

        post = get_object_or_404(Post, pk=post_id)

        parent = None
        if parent_id:
            parent = get_object_or_404(Comment, pk=parent_id, post=post)

        comment = Comment.objects.create(
            post=post,
            author=request.user,
            parent=parent,
            body=body,
        )

        return JsonResponse({
            "status": "ok",
            "comment": {
                "id": comment.pk,
                "body": comment.body,
                "author": comment.author.username,
                "author_role": comment.author.role,
                "is_expert": comment.is_expert_response,
                "is_company": comment.is_company_response,
                "created_at": comment.created_at.strftime("%b %d, %Y %H:%M"),
                "parent_id": parent_id,
            },
        })
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)


# ============================================================================
# AJAX: Toggle Like
# ============================================================================

@login_required
@require_POST
def api_toggle_like(request, post_id):
    """AJAX toggle like on a post."""
    post = get_object_or_404(Post, pk=post_id)

    existing = PostLike.objects.filter(post=post, user=request.user)
    if existing.exists():
        existing.delete()
        liked = False
    else:
        PostLike.objects.create(post=post, user=request.user, reaction_type="like")
        liked = True

    return JsonResponse({
        "status": "ok",
        "liked": liked,
        "count": post.likes.count(),
    })


# ============================================================================
# AJAX: Toggle Comment Like
# ============================================================================

@login_required
@require_POST
def api_toggle_comment_like(request, comment_id):
    """AJAX toggle like on a comment."""
    comment = get_object_or_404(Comment, pk=comment_id)

    existing = CommentLike.objects.filter(comment=comment, user=request.user)
    if existing.exists():
        existing.delete()
        liked = False
    else:
        CommentLike.objects.create(comment=comment, user=request.user)
        liked = True

    return JsonResponse({
        "status": "ok",
        "liked": liked,
        "count": comment.likes.count(),
    })


# ============================================================================
# AJAX: Toggle Bookmark
# ============================================================================

@login_required
@require_POST
def api_toggle_bookmark(request, post_id):
    """AJAX toggle bookmark on a post."""
    post = get_object_or_404(Post, pk=post_id)

    existing = Bookmark.objects.filter(post=post, user=request.user)
    if existing.exists():
        existing.delete()
        bookmarked = False
    else:
        Bookmark.objects.create(post=post, user=request.user)
        bookmarked = True

    return JsonResponse({
        "status": "ok",
        "bookmarked": bookmarked,
    })


# ============================================================================
# AJAX: Accept Answer
# ============================================================================

@login_required
@require_POST
def api_accept_answer(request, comment_id):
    """Mark a comment as accepted answer (post author only)."""
    comment = get_object_or_404(Comment, pk=comment_id)

    if comment.post.author != request.user:
        return JsonResponse({"error": "Only the post author can accept answers."}, status=403)

    # Toggle
    comment.is_accepted_answer = not comment.is_accepted_answer
    comment.save(update_fields=["is_accepted_answer"])

    # If accepted, mark post as resolved
    if comment.is_accepted_answer:
        comment.post.status = "resolved"
        comment.post.save(update_fields=["status"])

    return JsonResponse({
        "status": "ok",
        "accepted": comment.is_accepted_answer,
    })


# ============================================================================
# AJAX: Recommendations
# ============================================================================

@login_required
def api_recommendations(request, post_id):
    """Get recommendations for a specific post."""
    post = get_object_or_404(Post, pk=post_id)

    from .recommendations import (
        recommend_companies_for_post,
        recommend_experts_for_post,
        recommend_similar_posts,
    )

    return JsonResponse({
        "status": "ok",
        "companies": recommend_companies_for_post(post),
        "experts": recommend_experts_for_post(post),
        "similar_posts": recommend_similar_posts(post),
    })


# ============================================================================
# AJAX: Edit Comment
# ============================================================================

@login_required
@require_POST
def api_edit_comment(request, comment_id):
    """AJAX endpoint to edit a comment (author only)."""
    comment = get_object_or_404(Comment, pk=comment_id)

    if comment.author != request.user:
        return JsonResponse({"error": "You can only edit your own comments."}, status=403)

    try:
        data = json.loads(request.body)
        body = data.get("body", "").strip()

        if not body:
            return JsonResponse({"error": "Comment cannot be empty."}, status=400)
        if len(body) > 5000:
            return JsonResponse({"error": "Comment too long."}, status=400)

        comment.body = body
        comment.save(update_fields=["body"])

        return JsonResponse({
            "status": "ok",
            "comment": {
                "id": comment.pk,
                "body": comment.body,
            },
        })
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)


# ============================================================================
# AJAX: Delete Comment
# ============================================================================

@login_required
@require_POST
def api_delete_comment(request, comment_id):
    """AJAX endpoint to delete a comment (author or post owner)."""
    comment = get_object_or_404(Comment, pk=comment_id)

    # Allow comment author OR post author to delete
    if comment.author != request.user and comment.post.author != request.user:
        return JsonResponse({"error": "Not authorized to delete this comment."}, status=403)

    comment.delete()
    return JsonResponse({"status": "ok"})


# ============================================================================
# AJAX: Report Post
# ============================================================================

@login_required
@require_POST
def api_report_post(request, post_id):
    """AJAX endpoint to report a post."""
    from .models import PostReport

    post = get_object_or_404(Post, pk=post_id)

    # Prevent duplicate reports
    if PostReport.objects.filter(post=post, reporter=request.user).exists():
        return JsonResponse({"error": "You've already reported this post."}, status=400)

    try:
        data = json.loads(request.body)
        reason = data.get("reason", "other")
        detail = data.get("detail", "").strip()

        valid_reasons = ["spam", "offensive", "misinformation", "other"]
        if reason not in valid_reasons:
            reason = "other"

        PostReport.objects.create(
            post=post,
            reporter=request.user,
            reason=reason,
            detail=detail,
        )

        return JsonResponse({"status": "ok", "message": "Report submitted. Thank you."})
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)


# ============================================================================
# Promote Post (Paid Boost)
# ============================================================================

@login_required
def promote_post(request, post_id):
    """
    Allow the post author to boost their post via simulated payment.

    Payment is SIMULATED — no real money moves. The user provides:
      - An account number / wallet ID (for reference only)
      - A 4-digit PIN   (hashed with SHA-256, never stored plain)
      - Duration (days) and budget amount

    On submission:
      - PromotedPost record is created with status='pending'
      - Admin must approve before the boost goes live
    """
    post = get_object_or_404(Post, pk=post_id, author=request.user)

    # Prevent double-promotion
    existing = PromotedPost.objects.filter(post=post).exclude(
        status__in=["expired", "rejected", "cancelled"]
    ).first()
    if existing:
        messages.warning(request, "This post already has an active or pending promotion.")
        return redirect("community:my_promotions")

    if request.method == "POST":
        # ── Collect form data ────────────────────────────────────────
        try:
            duration_days   = int(request.POST.get("duration_days", 7))
            amount          = float(request.POST.get("amount", 0))
            payment_method  = request.POST.get("payment_method", "balance")
            account_number  = request.POST.get("account_number", "").strip()
            pin_raw         = request.POST.get("pin", "").strip()
            target_role     = request.POST.get("target_role", "all")
            target_interest = request.POST.get("target_interest", "all")
            target_country  = request.POST.get("target_country", "").strip()
            target_city     = request.POST.get("target_city", "").strip()
        except (ValueError, TypeError):
            messages.error(request, "Invalid form data. Please check your inputs.")
            return redirect("community:promote_post", post_id=post_id)

        # ── Validation ───────────────────────────────────────────────
        errors = []
        if not (1 <= duration_days <= 30):
            errors.append("Duration must be between 1 and 30 days.")
        if amount < 100:
            errors.append("Minimum boost amount is 100 DA.")
        if not account_number:
            errors.append("Account number / wallet ID is required.")
        if not pin_raw or len(pin_raw) < 4:
            errors.append("PIN must be at least 4 digits.")
        valid_methods = [c[0] for c in PromotedPost._meta.get_field("payment_method").choices]
        if payment_method not in valid_methods:
            errors.append("Invalid payment method.")
        valid_roles = [c[0] for c in PromotedPost._meta.get_field("target_role").choices]
        if target_role not in valid_roles:
            errors.append("Invalid target role.")
        valid_interests = [c[0] for c in PromotedPost._meta.get_field("target_interest").choices]
        if target_interest not in valid_interests:
            errors.append("Invalid target interest.")

        if errors:
            for e in errors:
                messages.error(request, e)
            return render(request, "community/promote_post.html", {
                "post": post,
                "form_data": request.POST,
            })

        # ── Hash the PIN (SHA-256, never stored plain) ───────────────
        pin_hash = hashlib.sha256(pin_raw.encode("utf-8")).hexdigest()

        # ── Create promotion record & auto-activate immediately ──────
        promo = PromotedPost.objects.create(
            post            = post,
            promoted_by     = request.user,
            duration_days   = duration_days,
            amount          = amount,
            payment_method  = payment_method,
            account_number  = account_number,
            pin_hash        = pin_hash,
            target_role     = target_role,
            target_interest = target_interest,
            target_country  = target_country,
            target_city     = target_city,
            status          = "pending",   # temporary; activate() will flip it
        )
        # Auto-activate — no admin approval required
        promo.activate()   # sets status=active, start_date, end_date, boost_score

        end_str = promo.end_date.strftime("%B %d, %Y") if promo.end_date else ""
        messages.success(
            request,
            f"🚀 Your post is now LIVE and boosted for {duration_days} day(s)! "
            f"Boost ends on {end_str}. Amount: {amount:.0f} DA via {promo.get_payment_method_display()}."
        )
        return redirect("community:post_detail", post_id=post.pk)

    # GET — show the form
    return render(request, "community/promote_post.html", {"post": post})


# ============================================================================
# My Promotions Dashboard
# ============================================================================

@login_required
def my_promotions(request):
    """User's promotion history and status dashboard."""
    now = timezone.now()
    # Auto-expire
    PromotedPost.objects.filter(
        promoted_by=request.user, status="active", end_date__lt=now
    ).update(status="expired")

    promotions = (
        PromotedPost.objects
        .filter(promoted_by=request.user)
        .select_related("post")
        .order_by("-created_at")
    )
    return render(request, "community/my_promotions.html", {
        "promotions": promotions,
    })


# ============================================================================
# Admin: Approve Promotion
# ============================================================================

@login_required
def admin_approve_promotion(request, promo_id):
    """Admin-only: approve and activate a pending promotion."""
    if not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect("community:feed")

    promo = get_object_or_404(PromotedPost, pk=promo_id, status="pending")
    promo.activate()
    messages.success(
        request,
        f"Promotion for '{promo.post.title[:40]}' approved and activated for "
        f"{promo.duration_days} day(s)."
    )
    return redirect("community:admin_promotions")


# ============================================================================
# Admin: Reject Promotion
# ============================================================================

@login_required
@require_POST
def admin_reject_promotion(request, promo_id):
    """Admin-only: reject a pending promotion with a reason."""
    if not request.user.is_staff:
        return JsonResponse({"error": "Access denied."}, status=403)

    promo = get_object_or_404(PromotedPost, pk=promo_id, status="pending")
    reason = request.POST.get("reason", "Rejected by admin.").strip()
    promo.status = "rejected"
    promo.rejected_reason = reason
    promo.save(update_fields=["status", "rejected_reason"])

    messages.warning(request, f"Promotion rejected: {reason}")
    return redirect("community:admin_promotions")


# ============================================================================
# Admin: Promotions Panel
# ============================================================================

@login_required
def admin_promotions(request):
    """Admin panel listing all promotions with filters."""
    if not request.user.is_staff:
        messages.error(request, "Access denied.")
        return redirect("community:feed")

    now = timezone.now()
    PromotedPost.objects.filter(status="active", end_date__lt=now).update(status="expired")

    status_filter = request.GET.get("status", "pending")
    promotions = PromotedPost.objects.select_related("post", "promoted_by")
    if status_filter in ("pending", "active", "expired", "rejected", "cancelled"):
        promotions = promotions.filter(status=status_filter)

    paginator = Paginator(promotions.order_by("-created_at"), 20)
    page = paginator.get_page(request.GET.get("page", 1))

    stats = {
        "pending":  PromotedPost.objects.filter(status="pending").count(),
        "active":   PromotedPost.objects.filter(status="active").count(),
        "expired":  PromotedPost.objects.filter(status="expired").count(),
        "rejected": PromotedPost.objects.filter(status="rejected").count(),
    }

    return render(request, "community/admin_promotions.html", {
        "promotions": page,
        "current_status": status_filter,
        "stats": stats,
    })


# ============================================================================
# AJAX: Track Promotion Click
# ============================================================================

@login_required
@require_POST
def api_promotion_click(request, post_id):
    """AJAX: Increment click counter when a user opens a promoted post."""
    try:
        promo = PromotedPost.objects.get(post_id=post_id, status="active")
        PromotedPost.objects.filter(pk=promo.pk).update(clicks=F("clicks") + 1)
        return JsonResponse({"status": "ok"})
    except PromotedPost.DoesNotExist:
        return JsonResponse({"status": "ok"})  # Silently ignore if not promoted
