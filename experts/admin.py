# ============================================================================
# experts/admin.py
# Expert System — Django Admin Registration
# ============================================================================

from django.contrib import admin

from .models import Conversation, ExpertComplaint, ExpertProfile, Message, UserExpertRelation


@admin.register(ExpertProfile)
class ExpertProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "specialization", "is_active", "created_at")
    list_filter = ("is_active", "specialization")
    search_fields = ("user__username", "user__email", "specialization")


@admin.register(UserExpertRelation)
class UserExpertRelationAdmin(admin.ModelAdmin):
    list_display = ("user", "expert", "created_at")
    list_filter = ("created_at",)
    search_fields = ("user__username", "expert__username")


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("user", "expert", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("user__username", "expert__username")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("sender", "conversation", "is_read", "created_at")
    list_filter = ("is_read", "created_at")
    search_fields = ("sender__username", "text")


@admin.register(ExpertComplaint)
class ExpertComplaintAdmin(admin.ModelAdmin):
    list_display = ("expert", "title", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("expert__username", "title", "message")
