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
    path("history/", views.history, name="history"),

    # Auth
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),

    # File management
    path("save/", views.save_scan, name="save_scan"),
    path("my-files/", views.my_files, name="my_files"),
    path("scan/<str:photo_id>/", views.scan_detail, name="scan_detail"),

    # Comparison
    path("compare/<str:photo_id>/", views.compare_scan, name="compare_scan"),
    path("compare-ajax/<str:photo_id>/", views.compare_ajax, name="compare_ajax"),
    path("save-followup/<str:photo_id>/", views.save_followup, name="save_followup"),
]