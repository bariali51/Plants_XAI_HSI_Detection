# ============================================================================
# community/admin.py
# Agricultural Social Platform — Community Admin
# ============================================================================

from django.contrib import admin
from .models import (
    Bookmark, Comment, CommentLike, Post, PostImage,
    PostLike, PostReport, Tag, PromotedPost,
)


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "slug", "created_at")
    list_filter = ("category",)
    search_fields = ("name",)
    prepopulated_fields = {"slug": ("name",)}


class PostImageInline(admin.TabularInline):
    model = PostImage
    extra = 0


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ("title", "author", "post_type", "status", "is_ad", "view_count", "created_at")
    list_filter = ("post_type", "status", "is_ad", "is_pinned", "created_at")
    search_fields = ("title", "body", "author__username")
    filter_horizontal = ("tags",)
    inlines = [PostImageInline]


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ("author", "post", "is_expert_response", "is_company_response", "is_accepted_answer", "created_at")
    list_filter = ("is_expert_response", "is_company_response", "is_accepted_answer")
    search_fields = ("body", "author__username")


@admin.register(PostLike)
class PostLikeAdmin(admin.ModelAdmin):
    list_display = ("user", "post", "reaction_type", "created_at")
    list_filter = ("reaction_type",)


@admin.register(CommentLike)
class CommentLikeAdmin(admin.ModelAdmin):
    list_display = ("user", "comment", "created_at")


@admin.register(PostReport)
class PostReportAdmin(admin.ModelAdmin):
    list_display = ("post", "reporter", "reason", "status", "created_at")
    list_filter = ("reason", "status")
    search_fields = ("post__title", "reporter__username")


@admin.register(Bookmark)
class BookmarkAdmin(admin.ModelAdmin):
    list_display = ("user", "post", "created_at")
    search_fields = ("user__username", "post__title")


@admin.register(PromotedPost)
class PromotedPostAdmin(admin.ModelAdmin):
    list_display = (
        "post", "promoted_by", "status", "duration_days", "amount",
        "payment_method", "target_role", "target_interest",
        "boost_score", "impressions", "clicks", "created_at",
    )
    list_filter  = ("status", "payment_method", "target_role", "target_interest")
    search_fields = ("post__title", "promoted_by__username", "account_number")
    readonly_fields = ("boost_score", "impressions", "clicks", "pin_hash", "created_at", "updated_at")
    actions = ["approve_selected", "reject_selected"]

    def approve_selected(self, request, queryset):
        count = 0
        for promo in queryset.filter(status="pending"):
            promo.activate()
            count += 1
        self.message_user(request, f"{count} promotion(s) approved and activated.")
    approve_selected.short_description = "✅ Approve selected promotions"

    def reject_selected(self, request, queryset):
        updated = queryset.filter(status="pending").update(
            status="rejected",
            rejected_reason="Bulk rejected via admin panel.",
        )
        self.message_user(request, f"{updated} promotion(s) rejected.")
    reject_selected.short_description = "❌ Reject selected promotions"
