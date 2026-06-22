# ============================================================================
# community/urls.py
# Agricultural Social Platform — Community URL Configuration
# ============================================================================

from django.urls import path
from . import views

app_name = "community"

urlpatterns = [
    # ── Feed ─────────────────────────────────────────────────────────
    path("", views.feed_view, name="feed"),

    # ── Post CRUD ────────────────────────────────────────────────────
    path("post/create/", views.create_post, name="create_post"),
    path("post/<int:post_id>/", views.post_detail, name="post_detail"),
    path("post/<int:post_id>/edit/", views.edit_post, name="edit_post"),
    path("post/<int:post_id>/delete/", views.delete_post, name="delete_post"),

    # ── Filtering ────────────────────────────────────────────────────
    path("search/", views.search_posts, name="search"),
    path("tag/<slug:tag_slug>/", views.posts_by_tag, name="posts_by_tag"),

    # ── User Profile ─────────────────────────────────────────────────
    path("profile/<int:user_id>/", views.user_profile, name="user_profile"),
    path("bookmarks/", views.my_bookmarks, name="bookmarks"),

    # ── Paid Promotion ───────────────────────────────────────────────
    path("post/<int:post_id>/promote/", views.promote_post, name="promote_post"),
    path("my-promotions/", views.my_promotions, name="my_promotions"),

    # ── Admin Promotion Panel ────────────────────────────────────────
    path("admin/promotions/", views.admin_promotions, name="admin_promotions"),
    path("admin/promotions/<int:promo_id>/approve/", views.admin_approve_promotion, name="admin_approve_promotion"),
    path("admin/promotions/<int:promo_id>/reject/", views.admin_reject_promotion, name="admin_reject_promotion"),

    # ── AJAX API ─────────────────────────────────────────────────────
    path("api/comment/", views.api_add_comment, name="api_add_comment"),
    path("api/comment/<int:comment_id>/edit/", views.api_edit_comment, name="api_edit_comment"),
    path("api/comment/<int:comment_id>/delete/", views.api_delete_comment, name="api_delete_comment"),
    path("api/like/<int:post_id>/", views.api_toggle_like, name="api_toggle_like"),
    path("api/comment-like/<int:comment_id>/", views.api_toggle_comment_like, name="api_toggle_comment_like"),
    path("api/bookmark/<int:post_id>/", views.api_toggle_bookmark, name="api_toggle_bookmark"),
    path("api/accept-answer/<int:comment_id>/", views.api_accept_answer, name="api_accept_answer"),
    path("api/report/<int:post_id>/", views.api_report_post, name="api_report_post"),
    path("api/recommendations/<int:post_id>/", views.api_recommendations, name="api_recommendations"),
    path("api/promotion-click/<int:post_id>/", views.api_promotion_click, name="api_promotion_click"),
]
