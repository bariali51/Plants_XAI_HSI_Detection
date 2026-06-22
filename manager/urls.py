# ============================================================================
# manager/urls.py
# Admin Dashboard URL Configuration
# ============================================================================

from django.urls import path

from . import views

app_name = "manager"

urlpatterns = [
    # Dashboard
    path("", views.admin_dashboard, name="dashboard"),

    # User management
    path("users/", views.admin_users, name="users"),
    path("users/<int:user_id>/", views.admin_user_detail, name="user_detail"),
    path("users/<int:user_id>/toggle-staff/", views.admin_toggle_staff, name="toggle_staff"),
    path("users/<int:user_id>/toggle-active/", views.admin_toggle_active, name="toggle_active"),
    path("users/<int:user_id>/delete/", views.admin_delete_user, name="delete_user"),

    # Scan management
    path("scans/", views.admin_scans, name="scans"),
    path("scans/<str:photo_id>/", views.admin_scan_detail, name="scan_detail"),
    path("scans/<str:photo_id>/delete/", views.admin_delete_scan, name="delete_scan"),

    # Audit log
    path("audit/", views.admin_audit_log, name="audit_log"),

    # Subscription management
    path("users/<int:user_id>/change-plan/", views.admin_manage_subscription, name="change_plan"),
    path("api/subscription-stats/", views.admin_subscription_stats, name="subscription_stats"),
    path("revenue/", views.admin_revenue_overview, name="revenue_overview"),

    # Complaint management (User)
    path("complaints/", views.admin_complaints, name="complaints"),
    path("complaints/<int:complaint_id>/", views.admin_complaint_detail, name="complaint_detail"),
    path("complaints/<int:complaint_id>/mark-seen/", views.admin_mark_complaint_seen, name="mark_complaint_seen"),

    # Expert management
    path("experts/", views.admin_experts_list, name="experts_list"),
    path("experts/create/", views.admin_create_expert, name="create_expert"),
    path("experts/<int:user_id>/", views.admin_expert_detail, name="expert_detail"),
    path("experts/<int:user_id>/assign/", views.admin_assign_expert, name="assign_expert"),
    path("experts/<int:user_id>/remove/<int:relation_id>/", views.admin_remove_expert_user, name="remove_expert_user"),
    path("experts/<int:user_id>/toggle-active/", views.admin_toggle_expert_active, name="toggle_expert_active"),
    path("experts/<int:user_id>/delete/", views.admin_delete_expert, name="delete_expert"),
    path("experts/<int:user_id>/edit-profile/", views.admin_edit_expert_profile, name="edit_expert_profile"),

    # Admin scan upload
    path("upload-scan/", views.admin_upload_scan, name="upload_scan"),

    # Expert complaints (Admin)
    path("expert-complaints/", views.admin_expert_complaints, name="expert_complaints"),
    path("expert-complaints/<int:complaint_id>/", views.admin_expert_complaint_detail, name="expert_complaint_detail"),
    path("expert-complaints/<int:complaint_id>/mark-seen/", views.admin_mark_expert_complaint_seen, name="mark_expert_complaint_seen"),

    # Admin ↔ Expert Messaging
    path("messages/", views.admin_messages_inbox, name="admin_messages"),
    path("messages/<int:conversation_id>/", views.admin_conversation_view, name="admin_conversation"),
    path("messages/start/<int:expert_id>/", views.admin_start_conversation, name="admin_start_conversation"),
    path("api/admin-send-message/", views.api_admin_send_message, name="api_admin_send_message"),
    path("api/admin-messages/<int:conversation_id>/", views.api_admin_get_messages, name="api_admin_get_messages"),
    path("api/admin-react/<int:message_id>/", views.api_admin_react_message, name="api_admin_react_message"),
    path("api/admin-delete-message/<int:message_id>/", views.api_admin_delete_message, name="api_admin_delete_message"),
]
