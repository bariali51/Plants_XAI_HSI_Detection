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


ROLE_CHOICES = [
    ("user", "User"),
    ("expert", "Expert"),
    ("company", "Company"),
    ("admin", "Admin"),
]

PLAN_CHOICES = [
    ("free", "Free"),
    ("basic", "Basic"),
    ("premium", "Premium"),
]

PLAN_CONFIG = {
    "free": {
        "label": "Free",
        "price": 0,
        "currency": "DA",
        "daily_limit": 3,
        "chat_enabled": False,
        "description": "Basic access with limited daily scans",
    },
    "basic": {
        "label": "Basic",
        "price": 1000,
        "currency": "DA",
        "daily_limit": 50,
        "chat_enabled": False,
        "description": "Extended scanning with 50 scans per day",
    },
    "premium": {
        "label": "Premium",
        "price": 2500,
        "currency": "DA",
        "daily_limit": None,  # Unlimited
        "chat_enabled": True,
        "description": "Unlimited scans and full AI Assistant access",
    },
}


class CustomUser(AbstractUser):
    """Extended user model with additional profile fields."""

    email = models.EmailField(unique=True)
    email_verified = models.BooleanField(default=False)
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

    # ── Social Profile Fields ────────────────────────────────────────
    avatar = models.ImageField(upload_to="avatars/", null=True, blank=True)
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=100, blank=True)
    crops_of_interest = models.JSONField(default=list, blank=True)

    # ── Role ─────────────────────────────────────────────────────────
    role = models.CharField(
        max_length=10,
        choices=ROLE_CHOICES,
        default="user",
    )

    # ── Subscription Fields ──────────────────────────────────────────
    plan_type = models.CharField(
        max_length=10,
        choices=PLAN_CHOICES,
        default="free",
    )
    subscription_start = models.DateTimeField(null=True, blank=True)
    subscription_end = models.DateTimeField(null=True, blank=True)
    daily_scan_count = models.PositiveIntegerField(default=0)
    last_scan_date = models.DateField(null=True, blank=True)
    total_scans_count = models.PositiveIntegerField(default=0)

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"

    def __str__(self):
        return self.username

    def get_display_name(self):
        """Return name based on role: company name for companies, full name for others."""
        if self.role == "company":
            try:
                return self.company_profile.company_name
            except Exception:
                return self.get_full_name() or self.username
        return self.get_full_name() or self.username

    # ── Role Helpers ──────────────────────────────────────────────────

    @property
    def is_expert(self):
        """Check if user has the expert role."""
        return self.role == "expert"

    @property
    def is_company(self):
        """Check if user has the company role."""
        return self.role == "company"

    @property
    def is_admin_role(self):
        """Check if user has admin role or is Django staff."""
        return self.role == "admin" or self.is_staff

    # ── Plan Helpers ─────────────────────────────────────────────────

    @property
    def effective_plan_type(self):
        """Return effective plan: staff always get premium."""
        if self.is_staff:
            return "premium"
        return self.plan_type

    @property
    def plan_config(self):
        """Return the configuration dict for the user's effective plan."""
        return PLAN_CONFIG.get(self.effective_plan_type, PLAN_CONFIG["free"])

    def get_daily_limit(self):
        """Return the daily scan limit (None = unlimited)."""
        return self.plan_config["daily_limit"]

    def remaining_scans_today(self):
        """Return remaining scans for today (None = unlimited)."""
        from django.utils import timezone as tz
        self._reset_daily_count_if_needed(tz.localdate())
        limit = self.get_daily_limit()
        if limit is None:
            return None
        return max(0, limit - self.daily_scan_count)

    def can_scan(self):
        """Check if the user is allowed to perform a scan right now."""
        if self.is_staff:
            return True
        from django.utils import timezone as tz
        self._reset_daily_count_if_needed(tz.localdate())
        limit = self.get_daily_limit()
        if limit is None:
            return True
        return self.daily_scan_count < limit

    def increment_scan_count(self):
        """Record a scan: increment daily + total counters."""
        from django.utils import timezone as tz
        today = tz.localdate()
        self._reset_daily_count_if_needed(today)
        self.daily_scan_count += 1
        self.total_scans_count += 1
        self.last_scan_date = today
        self.save(update_fields=[
            "daily_scan_count", "total_scans_count", "last_scan_date",
        ])

    def can_use_chat(self):
        """Premium users and staff can access the AI Assistant chat."""
        if self.is_staff:
            return True
        return self.plan_type == "premium"

    def is_subscription_active(self):
        """Check if a paid subscription is currently active."""
        from django.utils import timezone as tz
        if self.plan_type == "free":
            return True  # Free is always "active"
        if self.subscription_end is None:
            return True
        return tz.now() < self.subscription_end

    def check_and_expire_subscription(self):
        """If subscription has expired, downgrade the user to free."""
        if self.is_staff:
            return False  # Staff never expire
        if self.plan_type == "free":
            return False
        if not self.is_subscription_active():
            self.plan_type = "free"
            self.subscription_start = None
            self.subscription_end = None
            self.save(update_fields=[
                "plan_type", "subscription_start", "subscription_end",
            ])
            return True  # was expired
        return False

    def _reset_daily_count_if_needed(self, today):
        """Reset daily scan count if the last scan was on a different day."""
        if self.last_scan_date != today:
            self.daily_scan_count = 0
            self.last_scan_date = today
            self.save(update_fields=["daily_scan_count", "last_scan_date"])


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
    ai_medical = models.TextField(null=True, blank=True)
    ai_treatment = models.TextField(null=True, blank=True)
    ai_irrigation = models.TextField(null=True, blank=True)
    ai_economic = models.TextField(null=True, blank=True)
    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Follow-Up Scan"
        verbose_name_plural = "Follow-Up Scans"

    def __str__(self):
        return f"Follow-up {self.follow_id} for {self.parent_scan.photo_id}"


# ============================================================================
# Fake Payment (Demo / Testing)
# ============================================================================

PAYMENT_STATUS_CHOICES = [
    ("pending", "Pending"),
    ("approved", "Approved"),
    ("rejected", "Rejected"),
]


class FakePayment(models.Model):
    """Demo payment record for plan upgrades — NOT a real payment system."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="fake_payments",
    )
    full_name = models.CharField(max_length=150)
    wallet_number = models.CharField(max_length=20)
    plan_requested = models.CharField(
        max_length=10,
        choices=[("basic", "Basic"), ("premium", "Premium")],
    )
    status = models.CharField(
        max_length=10,
        choices=PAYMENT_STATUS_CHOICES,
        default="pending",
    )
    amount = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    approved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Fake Payment"
        verbose_name_plural = "Fake Payments"

    def __str__(self):
        return f"{self.user.username} → {self.plan_requested} ({self.status})"


# ============================================================================
# Complaint (Shkawi) System
# ============================================================================

COMPLAINT_STATUS_CHOICES = [
    ("unseen", "Unseen"),
    ("seen", "Seen"),
]


class Complaint(models.Model):
    """User complaint / feedback system."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="complaints",
    )
    title = models.CharField(max_length=200)
    message = models.TextField()
    status = models.CharField(
        max_length=10,
        choices=COMPLAINT_STATUS_CHOICES,
        default="unseen",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Complaint"
        verbose_name_plural = "Complaints"

    def __str__(self):
        return f"[{self.status}] {self.title} — {self.user.username}"


# ============================================================================
# AI Chat Sessions (Multi-Conversation System)
# ============================================================================

class ChatSession(models.Model):
    """Individual AI chat conversation (like ChatGPT sessions)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
    )
    title = models.CharField(max_length=200, default="New Chat")
    is_active = models.BooleanField(default=True)
    is_closed = models.BooleanField(default=False)
    message_count = models.PositiveIntegerField(default=0)
    last_message_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Chat Session"
        verbose_name_plural = "Chat Sessions"

    def __str__(self):
        return f"{self.title} — {self.user.username}"

    @property
    def has_messages(self):
        """Check if the session has any messages."""
        return self.message_count > 0


class ChatMessage(models.Model):
    """Individual message in a ChatSession."""

    ROLE_CHOICES = [
        ("user", "User"),
        ("ai", "AI"),
    ]

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=5, choices=ROLE_CHOICES)
    content = models.TextField()
    is_offline = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"

    def __str__(self):
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"[{self.role}] {preview}"


# ============================================================================
# Email Verification Code (6-Digit)
# ============================================================================

class EmailVerificationCode(models.Model):
    """6-digit email verification code for account activation."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="verification_codes",
    )
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    is_used = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Email Verification Code"
        verbose_name_plural = "Email Verification Codes"

    def __str__(self):
        return f"{self.user.username} — {self.code}"

    def is_expired(self):
        """Check if code has expired (15 minutes)."""
        from django.utils import timezone
        from datetime import timedelta
        return timezone.now() > self.created_at + timedelta(minutes=15)


# ============================================================================
# Company
# ============================================================================

class Company(models.Model):
    """Company entity for company-type users."""

    name = models.CharField(max_length=200)
    email = models.EmailField(blank=True)
    description = models.TextField(blank=True)
    website = models.URLField(max_length=300, blank=True)
    phone = models.CharField(max_length=30, blank=True)
    address = models.TextField(blank=True)
    logo = models.ImageField(upload_to="companies/logos/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Company"
        verbose_name_plural = "Companies"

    def __str__(self):
        return self.name