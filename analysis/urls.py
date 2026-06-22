from django.urls import path
from . import views

urlpatterns = [
    # Main Dashboard
    path("", views.home, name="home"),

    # Analysis pages
    path("progress/", views.progress_view, name="progress"),
    path("treatment/", views.treatment_view, name="treatment"),

    # Info pages
    path("model/", views.model_info, name="model"),
    # history removed (#8)

    # AI Assistant
    path("assistant/", views.assistant_view, name="assistant"),

    # Auth
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),

    # File management
    path("save/", views.save_scan, name="save_scan"),
    path("my-files/", views.my_files, name="my_files"),
    path("scan/<str:photo_id>/", views.scan_detail, name="scan_detail"),
    path("scan/<str:photo_id>/delete/", views.delete_scan_file, name="delete_scan_file"),

    # Comparison
    path("compare/<str:photo_id>/", views.compare_scan, name="compare_scan"),
    path("compare-ajax/<str:photo_id>/", views.compare_ajax, name="compare_ajax"),
    path("save-followup/<str:photo_id>/", views.save_followup, name="save_followup"),

    # AI API Endpoints
    path("api/ai/chat/", views.ai_chat, name="ai_chat"),
    path("api/ai/summarize/<str:photo_id>/", views.ai_summarize, name="ai_summarize"),
    path("api/ai/treatment-plan/", views.ai_treatment_plan, name="ai_treatment_plan"),
    path("api/ai/recommendations/", views.ai_recommendations, name="ai_recommendations"),

    path("save-followupnew/<str:photo_id>/", views.save_followupnew, name="save_followupnew"),

    # Clear last analysis session
    path("api/clear-last-analysis/", views.clear_last_analysis, name="clear_last_analysis"),

    # Subscription
    path("api/subscription/status/", views.subscription_status, name="subscription_status"),
    path("plans/", views.plans_page, name="plans"),
    path("upgrade/", views.upgrade_plan, name="upgrade_plan"),

    # Settings
    path("settings/", views.settings_page, name="settings"),

    # Complaints (Shkawi)
    path("complaints/", views.my_complaints, name="my_complaints"),
    path("complaints/<int:complaint_id>/", views.complaint_detail, name="complaint_detail"),
    path("complaints/<int:complaint_id>/delete/", views.delete_complaint, name="delete_complaint"),

    # Email Verification
    path("verify-code/", views.verify_code_view, name="verify_code"),
    path("verify-email/<str:token>/", views.verify_email_view, name="verify_email"),
    path("resend-verification/", views.resend_verification, name="resend_verification"),

    # Chat Sessions (Multi-Conversation API)
    path("api/chat/sessions/", views.chat_sessions_list, name="chat_sessions"),
    path("api/chat/sessions/create/", views.chat_session_create, name="chat_session_create"),
    path("api/chat/sessions/<int:session_id>/messages/", views.chat_session_messages, name="chat_session_messages"),
    path("api/chat/sessions/<int:session_id>/send/", views.chat_session_send, name="chat_session_send"),
    path("api/chat/sessions/<int:session_id>/delete/", views.chat_session_delete, name="chat_session_delete"),
    path("api/chat/sessions/<int:session_id>/close/", views.chat_session_close, name="chat_session_close"),
]