# ============================================================================
# experts/urls.py
# Expert System — URL Configuration
# ============================================================================

from django.urls import path

from . import views

app_name = "experts"

urlpatterns = [
    # ── Expert Dashboard ─────────────────────────────────────────────
    path("dashboard/", views.expert_dashboard, name="dashboard"),

    # ── Expert Messages ──────────────────────────────────────────────
    path("messages/", views.expert_messages, name="messages"),
    path("messages/<int:conversation_id>/", views.expert_conversation, name="conversation"),

    # ── Expert Features ──────────────────────────────────────────────
    path("my-files/", views.expert_my_files, name="my_files"),
    path("complaints/", views.expert_complaints, name="complaints"),
    path("complaints/<int:complaint_id>/delete/", views.expert_delete_complaint, name="delete_complaint"),
    path("settings/", views.expert_settings, name="settings"),

    # ── User-Facing Expert Features ──────────────────────────────────
    path("my-experts/", views.user_experts, name="user_experts"),
    path("chat/<int:conversation_id>/", views.user_conversation, name="user_conversation"),

    # ── API (AJAX) ───────────────────────────────────────────────────
    path("api/send-message/", views.api_send_message, name="api_send_message"),
    path("api/messages/<int:conversation_id>/", views.api_get_messages, name="api_get_messages"),
    path("api/mark-read/<int:conversation_id>/", views.api_mark_read, name="api_mark_read"),
    path("api/delete-message/<int:message_id>/", views.api_delete_message, name="api_delete_message"),
    path("api/react/<int:message_id>/", views.api_toggle_reaction, name="api_toggle_reaction"),

    # ── Admin Messages (Expert Side) ─────────────────────────────────
    path("admin-messages/", views.expert_admin_messages, name="admin_messages"),
    path("admin-messages/<int:conversation_id>/", views.expert_admin_conversation, name="admin_conversation"),
    path("api/admin-reply/", views.api_expert_admin_reply, name="api_expert_admin_reply"),
    path("api/admin-poll/<int:conversation_id>/", views.api_expert_admin_poll, name="api_expert_admin_poll"),
    path("api/admin-react/<int:message_id>/", views.api_expert_admin_react, name="api_expert_admin_react"),
    path("api/admin-delete-msg/<int:message_id>/", views.api_expert_admin_delete_msg, name="api_expert_admin_delete_msg"),

    # ── Public Profile ───────────────────────────────────────────────
    path("profile/<int:user_id>/", views.public_expert_profile, name="public_profile"),
]
