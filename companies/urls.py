# ============================================================================
# companies/urls.py
# Agricultural Social Platform — Company URL Configuration
# ============================================================================

from django.urls import path
from . import views

app_name = "companies"

urlpatterns = [
    # ── Directory & Profiles ─────────────────────────────────────────
    path("", views.company_directory, name="directory"),
    path("dashboard/", views.company_dashboard, name="dashboard"),
    path("<int:company_id>/", views.company_profile, name="profile"),
    path("<int:company_id>/review/", views.add_review, name="add_review"),

    # ── Product & Service Management ─────────────────────────────────
    path("products/", views.manage_products, name="manage_products"),
    path("products/<int:product_id>/delete/", views.delete_product, name="delete_product"),
    path("services/", views.manage_services, name="manage_services"),
    path("services/<int:service_id>/delete/", views.delete_service, name="delete_service"),

    # ── User↔Company Messaging (Company Side) ───────────────────────
    path("messages/", views.company_messages_inbox, name="company_messages"),
    path("messages/<int:conversation_id>/", views.company_conversation_view, name="company_conversation"),

    # ── User↔Company Messaging (User Side) ───────────────────────────
    path("my-messages/", views.user_company_messages, name="user_messages"),
    path("my-messages/<int:conversation_id>/", views.user_company_conversation, name="user_conversation"),
    path("<int:company_id>/start-chat/", views.start_company_chat, name="start_chat"),

    # ── Messaging API (AJAX) ─────────────────────────────────────────
    path("api/send-message/", views.api_company_send_message, name="api_send_message"),
    path("api/messages/<int:conversation_id>/", views.api_company_get_messages, name="api_get_messages"),
    path("api/mark-read/<int:conversation_id>/", views.api_company_mark_read, name="api_mark_read"),
    path("api/delete-message/<int:message_id>/", views.api_delete_message, name="api_delete_message"),

    # ── Search API ───────────────────────────────────────────────────
    path("api/search/", views.api_company_search, name="api_search"),
]
