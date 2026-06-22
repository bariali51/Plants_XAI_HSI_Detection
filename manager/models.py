# ============================================================================
# manager/models.py
# Admin Dashboard — Audit Logging Model
# ============================================================================

from django.conf import settings
from django.db import models


class AuditLog(models.Model):
    """Records admin/manager actions for accountability and traceability."""

    ACTION_CHOICES = [
        ("view_dashboard", "Viewed Dashboard"),
        ("view_user", "Viewed User"),
        ("toggle_staff", "Toggled Staff Status"),
        ("deactivate_user", "Deactivated User"),
        ("activate_user", "Activated User"),
        ("delete_scan", "Deleted Scan"),
        ("delete_user", "Deleted User"),
        ("delete_expert", "Deleted Expert"),
        ("view_scan", "Viewed Scan"),
        ("export_data", "Exported Data"),
        ("change_plan", "Changed User Plan"),
        ("create_expert", "Created Expert Account"),
        ("assign_expert", "Assigned User to Expert"),
        ("remove_expert_user", "Removed User from Expert"),
        ("toggle_expert", "Toggled Expert Status"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="audit_logs",
    )
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    target_type = models.CharField(max_length=50, blank=True)
    target_id = models.CharField(max_length=100, blank=True)
    detail = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Audit Log"
        verbose_name_plural = "Audit Logs"

    def __str__(self):
        return f"[{self.action}] by {self.user} at {self.created_at:%Y-%m-%d %H:%M}"
