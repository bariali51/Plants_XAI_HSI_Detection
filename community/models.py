# ============================================================================
# community/models.py
# Agricultural Social Platform — Community Models
# ============================================================================

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


# ============================================================================
# Tag (Shared taxonomy for crops, diseases, topics, services)
# ============================================================================

TAG_CATEGORY_CHOICES = [
    ("crop", "Crop"),
    ("disease", "Disease"),
    ("topic", "Topic"),
    ("service", "Service"),
]


class Tag(models.Model):
    """Reusable tag for categorizing posts, products, services, and companies."""

    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=120, unique=True, blank=True)
    category = models.CharField(
        max_length=20,
        choices=TAG_CATEGORY_CHOICES,
        default="topic",
    )
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Tag"
        verbose_name_plural = "Tags"

    def __str__(self):
        return f"{self.name} ({self.get_category_display()})"

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


# ============================================================================
# Post
# ============================================================================

POST_TYPE_CHOICES = [
    ("question", "Question"),
    ("solution", "Solution"),
    ("discussion", "Discussion"),
    ("alert", "Alert"),
]

POST_STATUS_CHOICES = [
    ("active", "Active"),
    ("resolved", "Resolved"),
    ("closed", "Closed"),
]


class Post(models.Model):
    """Community post about an agricultural problem, question, or solution."""

    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_posts",
    )
    title = models.CharField(max_length=300)
    body = models.TextField()

    post_type = models.CharField(
        max_length=20,
        choices=POST_TYPE_CHOICES,
        default="question",
    )
    status = models.CharField(
        max_length=20,
        choices=POST_STATUS_CHOICES,
        default="active",
    )

    tags = models.ManyToManyField(Tag, blank=True, related_name="posts")

    is_pinned = models.BooleanField(default=False)
    is_ad = models.BooleanField(
        default=False,
        help_text="True when this post has an active paid promotion (boosted/sponsored).",
    )
    view_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_pinned", "-created_at"]
        verbose_name = "Post"
        verbose_name_plural = "Posts"
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["author", "-created_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["post_type"]),
            models.Index(fields=["is_ad"]),
        ]

    def __str__(self):
        return self.title

    @property
    def like_count(self):
        return self.likes.count()

    @property
    def comment_count(self):
        return self.comments.count()

    @property
    def author_role(self):
        return self.author.role


# ============================================================================
# Post Image (Multiple images per post)
# ============================================================================

class PostImage(models.Model):
    """Image attachment for a post (up to 4 per post)."""

    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name="images",
    )
    image = models.ImageField(upload_to="community/posts/")
    caption = models.CharField(max_length=200, blank=True)
    order = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["order"]
        verbose_name = "Post Image"
        verbose_name_plural = "Post Images"

    def __str__(self):
        return f"Image {self.order} for: {self.post.title[:40]}"


# ============================================================================
# Comment (Threaded)
# ============================================================================

class Comment(models.Model):
    """Threaded comment on a post. Supports nesting via parent FK."""

    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name="comments",
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_comments",
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="replies",
    )
    body = models.TextField()

    is_expert_response = models.BooleanField(default=False)
    is_company_response = models.BooleanField(default=False)
    is_accepted_answer = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_accepted_answer", "-is_expert_response", "created_at"]
        verbose_name = "Comment"
        verbose_name_plural = "Comments"
        indexes = [
            models.Index(fields=["post", "created_at"]),
        ]

    def __str__(self):
        preview = self.body[:50] + "..." if len(self.body) > 50 else self.body
        return f"{self.author.username}: {preview}"

    def save(self, *args, **kwargs):
        """Auto-detect expert/company responses based on author role."""
        if self.author.role == "expert":
            self.is_expert_response = True
        elif self.author.role == "company":
            self.is_company_response = True
        super().save(*args, **kwargs)

    @property
    def like_count(self):
        return self.likes.count()

    @property
    def reply_count(self):
        return self.replies.count()


# ============================================================================
# Post Like / Reaction
# ============================================================================

REACTION_TYPE_CHOICES = [
    ("like", "Like"),
    ("helpful", "Helpful"),
    ("insightful", "Insightful"),
]


class PostLike(models.Model):
    """Like or reaction on a post."""

    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name="likes",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="post_likes",
    )
    reaction_type = models.CharField(
        max_length=20,
        choices=REACTION_TYPE_CHOICES,
        default="like",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("post", "user")
        verbose_name = "Post Like"
        verbose_name_plural = "Post Likes"

    def __str__(self):
        return f"{self.user.username} → {self.reaction_type} on {self.post.title[:30]}"


# ============================================================================
# Comment Like
# ============================================================================

class CommentLike(models.Model):
    """Like on a comment."""

    comment = models.ForeignKey(
        Comment,
        on_delete=models.CASCADE,
        related_name="likes",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="comment_likes",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("comment", "user")
        verbose_name = "Comment Like"
        verbose_name_plural = "Comment Likes"

    def __str__(self):
        return f"{self.user.username} liked comment #{self.comment.pk}"


# ============================================================================
# Post Report (Flagging system)
# ============================================================================

REPORT_REASON_CHOICES = [
    ("spam", "Spam"),
    ("inappropriate", "Inappropriate Content"),
    ("misinformation", "Misinformation"),
    ("harassment", "Harassment"),
    ("other", "Other"),
]

REPORT_STATUS_CHOICES = [
    ("pending", "Pending"),
    ("reviewed", "Reviewed"),
    ("dismissed", "Dismissed"),
]


class PostReport(models.Model):
    """Report/flag on a post for moderation."""

    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name="reports",
    )
    reporter = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="post_reports",
    )
    reason = models.CharField(max_length=20, choices=REPORT_REASON_CHOICES)
    detail = models.TextField(blank=True)
    status = models.CharField(
        max_length=20,
        choices=REPORT_STATUS_CHOICES,
        default="pending",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("post", "reporter")
        ordering = ["-created_at"]
        verbose_name = "Post Report"
        verbose_name_plural = "Post Reports"

    def __str__(self):
        return f"Report on '{self.post.title[:30]}' by {self.reporter.username}"


# ============================================================================
# Bookmark
# ============================================================================

class Bookmark(models.Model):
    """User bookmarks a post for later reference."""

    post = models.ForeignKey(
        Post,
        on_delete=models.CASCADE,
        related_name="bookmarks",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="bookmarks",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("post", "user")
        ordering = ["-created_at"]
        verbose_name = "Bookmark"
        verbose_name_plural = "Bookmarks"

    def __str__(self):
        return f"{self.user.username} bookmarked '{self.post.title[:30]}'"


# ============================================================================
# Paid Post Promotion (Simulated Payment — No Real Card Required)
# ============================================================================

PROMOTION_STATUS_CHOICES = [
    ("pending",  "Pending Approval"),
    ("active",   "Active"),
    ("expired",  "Expired"),
    ("rejected", "Rejected"),
    ("cancelled","Cancelled"),
]

PAYMENT_METHOD_CHOICES = [
    ("balance", "Account Balance"),
    ("ccp",     "CCP / Postal Account"),
    ("baridimob", "BaridiMob"),
    ("dahabiya", "Dahabiya Card"),
]

TARGET_ROLE_CHOICES = [
    ("all",     "All Users"),
    ("farmer",  "Farmers Only"),
    ("company", "Companies Only"),
    ("expert",  "Experts Only"),
    ("user",    "General Users"),
]

INTEREST_CATEGORY_CHOICES = [
    ("all",         "All Interests"),
    ("pesticides",  "Medicines / Pesticides"),
    ("crops",       "Crops / Agriculture Products"),
    ("equipment",   "Equipment / Machinery"),
    ("fertilizers", "Fertilizers"),
    ("general",     "General Farming"),
]


class PromotedPost(models.Model):
    """
    Paid post promotion record.

    Stores:
    - Which post is boosted
    - Targeting rules (region, user role, interest)
    - Simulated payment info (account number + PIN — NOT a real gateway)
    - Duration & budget
    - Admin approval state
    - Real-time analytics (impressions, clicks)
    """

    # ── Core link ────────────────────────────────────────────────────
    post = models.OneToOneField(
        Post,
        on_delete=models.CASCADE,
        related_name="promotion",
    )
    promoted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="promotions",
    )

    # ── Promotion window ────────────────────────────────────────────
    duration_days = models.PositiveSmallIntegerField(
        default=7,
        help_text="Number of days to boost the post (1–30)",
    )
    start_date = models.DateTimeField(null=True, blank=True)
    end_date   = models.DateTimeField(null=True, blank=True)

    # ── Budget ───────────────────────────────────────────────────────
    amount = models.DecimalField(
        max_digits=10, decimal_places=2,
        help_text="Amount paid (DA or equivalent)",
    )
    payment_method = models.CharField(
        max_length=20,
        choices=PAYMENT_METHOD_CHOICES,
        default="balance",
    )
    # Simulated wallet/account reference — never stores real card numbers
    account_number = models.CharField(
        max_length=30,
        blank=True,
        help_text="Wallet ID or account number (simulated)",
    )
    # Simulated PIN — stored hashed, NOT plain text
    pin_hash = models.CharField(
        max_length=128,
        blank=True,
        help_text="SHA-256 hash of user-supplied PIN (never plain)",
    )

    # ── Targeting rules ──────────────────────────────────────────────
    target_country  = models.CharField(max_length=100, blank=True, default="")
    target_city     = models.CharField(max_length=100, blank=True, default="")
    target_role     = models.CharField(
        max_length=20,
        choices=TARGET_ROLE_CHOICES,
        default="all",
    )
    target_interest = models.CharField(
        max_length=30,
        choices=INTEREST_CATEGORY_CHOICES,
        default="all",
    )

    # ── Ranking boost ────────────────────────────────────────────────
    boost_score = models.FloatField(
        default=0.0,
        help_text="Computed boost weight used in feed ranking",
    )

    # ── Status & moderation ─────────────────────────────────────────
    status = models.CharField(
        max_length=20,
        choices=PROMOTION_STATUS_CHOICES,
        default="pending",
    )
    admin_note = models.TextField(blank=True)
    rejected_reason = models.TextField(blank=True)

    # ── Analytics ───────────────────────────────────────────────────
    impressions = models.PositiveIntegerField(default=0)
    clicks      = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering  = ["-created_at"]
        verbose_name = "Promoted Post"
        verbose_name_plural = "Promoted Posts"
        indexes = [
            models.Index(fields=["status", "end_date"]),
            models.Index(fields=["target_role"]),
            models.Index(fields=["target_country", "target_city"]),
        ]

    def __str__(self):
        return f"[{self.status.upper()}] '{self.post.title[:40]}' — {self.duration_days}d"

    # ── Computed properties ─────────────────────────────────────────
    @property
    def is_live(self):
        """True if the promotion is approved and within its time window."""
        now = timezone.now()
        return (
            self.status == "active"
            and self.start_date is not None
            and self.end_date is not None
            and self.start_date <= now <= self.end_date
        )

    @property
    def days_remaining(self):
        if not self.end_date:
            return 0
        remaining = (self.end_date - timezone.now()).days
        return max(remaining, 0)

    @property
    def ctr(self):
        """Click-through rate as a percentage."""
        if self.impressions == 0:
            return 0.0
        return round(self.clicks / self.impressions * 100, 2)

    def compute_boost_score(self):
        """
        Feed ranking boost score formula:
          boost = (amount × 0.5) + (duration_days × 0.3) + targeting_bonus

        Targeting bonuses reward precise targeting (better ROI for advertisers):
          +2.0  if targeting a specific role (not 'all')
          +1.5  if targeting a specific interest (not 'all')
          +1.0  if targeting a specific country
          +0.5  if targeting a specific city
        """
        targeting_bonus = 0.0
        if self.target_role != "all":     targeting_bonus += 2.0
        if self.target_interest != "all": targeting_bonus += 1.5
        if self.target_country:            targeting_bonus += 1.0
        if self.target_city:              targeting_bonus += 0.5

        self.boost_score = (
            float(self.amount) * 0.5
            + self.duration_days * 0.3
            + targeting_bonus
        )
        return self.boost_score

    def activate(self):
        """Approve and activate the promotion (called by admin)."""
        now = timezone.now()
        self.status     = "active"
        self.start_date = now
        self.end_date   = now + timezone.timedelta(days=self.duration_days)
        self.compute_boost_score()
        self.save(update_fields=["status", "start_date", "end_date", "boost_score"])
        # Mark the linked post as an ad
        self.post.is_ad = True
        self.post.save(update_fields=["is_ad"])

    def expire_if_due(self):
        """Mark expired if past end date."""
        if self.status == "active" and self.end_date and timezone.now() > self.end_date:
            self.status = "expired"
            self.save(update_fields=["status"])
            # Clear the ad flag on the linked post
            self.post.is_ad = False
            self.post.save(update_fields=["is_ad"])
            return True
        return False
