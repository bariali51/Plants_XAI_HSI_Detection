# ============================================================================
# analysis/models.py
# Plant Disease Detection — Django Models (Optimized)
# ============================================================================

import random
import uuid

from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


# ============================================================================
# Custom User
# ============================================================================

def generate_user_code():
    """Generate a random 6-digit user code."""
    return str(random.randint(100000, 999999))


class CustomUser(AbstractUser):
    """Extended user model with additional profile fields."""

    email = models.EmailField(unique=True)
    birth_date = models.DateField(null=True, blank=True)

    PURPOSE_CHOICES = [
        ("farmer", "Farmer"),
        ("company", "Company"),
        ("other", "Other"),
    ]

    purpose = models.CharField(
        max_length=20,
        choices=PURPOSE_CHOICES,
        default="other",
    )

    user_code = models.CharField(
        max_length=6,
        unique=True,
        default=generate_user_code,
    )

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    def __str__(self):
        return self.username


# ============================================================================
# Scan Result
# ============================================================================

def generate_photo_id():
    """Generate a unique 12-character photo ID."""
    return str(uuid.uuid4())[:12]


class ScanResult(models.Model):
    """Stores a single plant disease scan analysis."""

    photo_id = models.CharField(max_length=50, unique=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    image_original = models.ImageField(upload_to="scans/original/")
    image_gradcam = models.ImageField(upload_to="scans/gradcam/")

    prediction = models.CharField(max_length=120)
    confidence = models.FloatField()

    disease_ratio = models.FloatField()
    disease_stage = models.CharField(max_length=50)

    # AI Doctor Fields
    ai_medical = models.TextField(null=True, blank=True)
    ai_treatment = models.TextField(null=True, blank=True)
    ai_irrigation = models.TextField(null=True, blank=True)
    ai_economic = models.TextField(null=True, blank=True)

    yield_loss = models.FloatField(null=True, blank=True)
    fungicides_json = models.JSONField(null=True, blank=True)
    folder_name = models.CharField(max_length=120)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Scan Result"
        verbose_name_plural = "Scan Results"

    def __str__(self):
        return f"{self.prediction} ({self.photo_id})"


# ============================================================================
# Follow-Up Scan
# ============================================================================

class FollowUpScan(models.Model):
    """Stores follow-up scans for disease progression tracking."""

    follow_id = models.CharField(max_length=50, unique=True)

    parent_scan = models.ForeignKey(
        ScanResult,
        on_delete=models.CASCADE,
        related_name="progress_scans",
    )

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    new_image = models.ImageField(upload_to="followups/")
    new_gradcam = models.ImageField(upload_to="followups/")

    disease = models.CharField(max_length=120)
    confidence = models.FloatField()

    ratio = models.FloatField()
    stage = models.CharField(max_length=50)

    yield_loss = models.FloatField()
    evolution_text = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Follow-Up Scan"
        verbose_name_plural = "Follow-Up Scans"

    def __str__(self):
        return f"Follow-up {self.follow_id} for {self.parent_scan.photo_id}"