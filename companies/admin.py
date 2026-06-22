# ============================================================================
# companies/admin.py
# Agricultural Social Platform — Companies Admin
# ============================================================================

from django.contrib import admin
from .models import (
    CompanyProfile, Product, Service, CompanyReview,
    CompanyConversation, CompanyMessage,
)


@admin.register(CompanyProfile)
class CompanyProfileAdmin(admin.ModelAdmin):
    list_display = ("company_name", "user", "city", "verified", "is_active", "created_at")
    list_filter = ("verified", "is_active", "city")
    search_fields = ("company_name", "user__username", "specializations")


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("name", "company", "category", "price", "is_active", "created_at")
    list_filter = ("category", "is_active")
    search_fields = ("name", "company__company_name")


@admin.register(Service)
class ServiceAdmin(admin.ModelAdmin):
    list_display = ("name", "company", "service_area", "is_active", "created_at")
    list_filter = ("is_active",)
    search_fields = ("name", "company__company_name")


@admin.register(CompanyReview)
class CompanyReviewAdmin(admin.ModelAdmin):
    list_display = ("company", "user", "rating", "created_at")
    list_filter = ("rating",)
    search_fields = ("company__company_name", "user__username")


@admin.register(CompanyConversation)
class CompanyConversationAdmin(admin.ModelAdmin):
    list_display = ("user", "company", "subject", "is_active", "created_at", "updated_at")
    list_filter = ("is_active", "created_at")
    search_fields = ("user__username", "company__company_name")


@admin.register(CompanyMessage)
class CompanyMessageAdmin(admin.ModelAdmin):
    list_display = ("sender", "conversation", "is_read", "created_at")
    list_filter = ("is_read", "created_at")
    search_fields = ("sender__username", "text")
