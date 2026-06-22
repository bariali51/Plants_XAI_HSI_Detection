# ============================================================================
# companies/models.py
# Agricultural Social Platform — Company Models & Messaging
# ============================================================================

from django.conf import settings
from django.db import models


# ============================================================================
# Company Profile
# ============================================================================

class CompanyProfile(models.Model):
    """Extended company profile linked to a CustomUser with role='company'."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="company_profile",
    )
    company_name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    logo = models.ImageField(upload_to="companies/logos/", null=True, blank=True)
    cover_image = models.ImageField(upload_to="companies/covers/", null=True, blank=True)

    website = models.URLField(max_length=300, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=30, blank=True)
    address = models.TextField(blank=True)
    city = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)

    specializations = models.TextField(
        blank=True,
        help_text="Comma-separated areas of expertise",
    )
    tags = models.ManyToManyField(
        "community.Tag",
        blank=True,
        related_name="companies",
    )

    verified = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-verified", "-created_at"]
        verbose_name = "Company Profile"
        verbose_name_plural = "Company Profiles"

    def __str__(self):
        return self.company_name

    @property
    def logo_url(self):
        if self.logo:
            return self.logo.url
        return None

    @property
    def average_rating(self):
        reviews = self.reviews.all()
        if not reviews.exists():
            return 0
        return round(reviews.aggregate(models.Avg("rating"))["rating__avg"], 1)

    @property
    def review_count(self):
        return self.reviews.count()

    @property
    def product_count(self):
        return self.products.filter(is_active=True).count()

    @property
    def service_count(self):
        return self.services.filter(is_active=True).count()


# ============================================================================
# Product
# ============================================================================

PRODUCT_CATEGORY_CHOICES = [
    ("fertilizer", "Fertilizer"),
    ("pesticide", "Pesticide"),
    ("seed", "Seed"),
    ("equipment", "Equipment"),
    ("tool", "Tool"),
    ("supplement", "Supplement"),
    ("other", "Other"),
]


class Product(models.Model):
    """Agricultural product offered by a company."""

    company = models.ForeignKey(
        CompanyProfile,
        on_delete=models.CASCADE,
        related_name="products",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    currency = models.CharField(max_length=10, default="DA")
    image = models.ImageField(upload_to="companies/products/", null=True, blank=True)
    category = models.CharField(
        max_length=20,
        choices=PRODUCT_CATEGORY_CHOICES,
        default="other",
    )
    tags = models.ManyToManyField(
        "community.Tag",
        blank=True,
        related_name="products",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Product"
        verbose_name_plural = "Products"

    def __str__(self):
        return f"{self.name} — {self.company.company_name}"


# ============================================================================
# Service
# ============================================================================

class Service(models.Model):
    """Agricultural service offered by a company."""

    company = models.ForeignKey(
        CompanyProfile,
        on_delete=models.CASCADE,
        related_name="services",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    price_range = models.CharField(
        max_length=100,
        blank=True,
        help_text="e.g. '500-2000 DA' or 'Contact for quote'",
    )
    service_area = models.CharField(
        max_length=200,
        blank=True,
        help_text="Geographic area covered",
    )
    tags = models.ManyToManyField(
        "community.Tag",
        blank=True,
        related_name="services",
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Service"
        verbose_name_plural = "Services"

    def __str__(self):
        return f"{self.name} — {self.company.company_name}"


# ============================================================================
# Company Review
# ============================================================================

class CompanyReview(models.Model):
    """User review/rating for a company."""

    company = models.ForeignKey(
        CompanyProfile,
        on_delete=models.CASCADE,
        related_name="reviews",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="company_reviews",
    )
    rating = models.PositiveSmallIntegerField(
        help_text="Rating from 1 to 5",
    )
    review_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("company", "user")
        ordering = ["-created_at"]
        verbose_name = "Company Review"
        verbose_name_plural = "Company Reviews"

    def __str__(self):
        return f"{self.user.username} → {self.rating}★ for {self.company.company_name}"

    def clean(self):
        from django.core.exceptions import ValidationError
        if self.rating < 1 or self.rating > 5:
            raise ValidationError("Rating must be between 1 and 5.")


# ============================================================================
# User ↔ Company Conversation
# ============================================================================

class CompanyConversation(models.Model):
    """Private conversation thread between a user and a company."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="company_conversations",
    )
    company = models.ForeignKey(
        CompanyProfile,
        on_delete=models.CASCADE,
        related_name="conversations",
    )
    subject = models.CharField(max_length=200, default="General Inquiry")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "company")
        ordering = ["-updated_at"]
        verbose_name = "Company Conversation"
        verbose_name_plural = "Company Conversations"

    def __str__(self):
        return f"Chat: {self.user.username} ↔ {self.company.company_name}"

    @property
    def last_message(self):
        """Return the most recent message in the conversation."""
        return self.messages.order_by("-created_at").first()

    def unread_count_for(self, user):
        """Count unread messages for a specific user."""
        return self.messages.filter(is_read=False).exclude(sender=user).count()


# ============================================================================
# User ↔ Company Message
# ============================================================================

class CompanyMessage(models.Model):
    """Individual message in a user ↔ company conversation."""

    conversation = models.ForeignKey(
        CompanyConversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="company_sent_messages",
    )
    text = models.TextField(blank=True)
    image = models.ImageField(upload_to="companies/messages/images/", null=True, blank=True)
    file = models.FileField(upload_to="companies/messages/files/", null=True, blank=True)
    file_name = models.CharField(max_length=255, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Company Message"
        verbose_name_plural = "Company Messages"
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
        ]

    def __str__(self):
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"{self.sender.username}: {preview}"
