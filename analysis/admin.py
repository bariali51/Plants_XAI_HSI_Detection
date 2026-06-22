from django.contrib import admin

from .models import (
    ChatMessage,
    ChatSession,
    Company,
    Complaint,
    CustomUser,
    EmailVerificationCode,
    FakePayment,
    FollowUpScan,
    ScanResult,
)


@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ("username", "email", "role", "plan_type", "is_staff", "date_joined")
    list_filter = ("role", "plan_type", "is_staff", "is_active")
    search_fields = ("username", "email", "first_name", "last_name")


@admin.register(ScanResult)
class ScanResultAdmin(admin.ModelAdmin):
    list_display = ("photo_id", "user", "prediction", "confidence", "created_at")
    list_filter = ("prediction", "created_at")
    search_fields = ("photo_id", "user__username", "prediction")


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "is_active", "is_closed", "created_at", "updated_at")
    list_filter = ("is_active", "is_closed", "created_at")
    search_fields = ("title", "user__username")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ("session", "role", "is_offline", "created_at")
    list_filter = ("role", "is_offline")


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ("name", "email", "website", "created_at")
    search_fields = ("name", "email")


@admin.register(Complaint)
class ComplaintAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "status", "created_at")
    list_filter = ("status",)
    search_fields = ("title", "user__username")


@admin.register(FakePayment)
class FakePaymentAdmin(admin.ModelAdmin):
    list_display = ("user", "plan_requested", "amount", "status", "created_at")
    list_filter = ("status", "plan_requested")


@admin.register(FollowUpScan)
class FollowUpScanAdmin(admin.ModelAdmin):
    list_display = ("follow_id", "parent_scan", "disease", "confidence", "created_at")


@admin.register(EmailVerificationCode)
class EmailVerificationCodeAdmin(admin.ModelAdmin):
    list_display = ("user", "code", "created_at")
