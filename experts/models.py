# ============================================================================
# experts/models.py
# Expert System — Models
# ============================================================================

from django.conf import settings
from django.db import models

from analysis.models import COMPLAINT_STATUS_CHOICES


# ============================================================================
# Expert Profile
# ============================================================================

class ExpertProfile(models.Model):
    """Extended profile for expert users. Created only by admin."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="expert_profile",
    )
    specialization = models.CharField(max_length=200)
    title = models.CharField(max_length=100, blank=True, help_text="e.g. Senior Plant Pathologist")
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to="experts/avatars/", null=True, blank=True)
    skills = models.JSONField(default=list, blank=True, help_text="List of skill tags")
    linkedin_url = models.URLField(max_length=300, blank=True)
    website_url = models.URLField(max_length=300, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Expert Profile"
        verbose_name_plural = "Expert Profiles"

    def __str__(self):
        return f"Expert: {self.user.username} ({self.specialization})"

    @property
    def display_name(self):
        """Return full name or username."""
        full = f"{self.user.first_name} {self.user.last_name}".strip()
        return full or self.user.username

    @property
    def avatar_url(self):
        """Return avatar URL or None."""
        if self.avatar:
            return self.avatar.url
        return None

    @property
    def assigned_users_count(self):
        """Count of users assigned to this expert."""
        return self.user.user_relations.count()

    @property
    def can_accept_users(self):
        """Check if expert can accept more users (max 10)."""
        return self.assigned_users_count < 10


# ============================================================================
# User ↔ Expert Relationship
# ============================================================================

class UserExpertRelation(models.Model):
    """
    Links users to experts with enforced limits.

    Limits (enforced in service layer):
    - Each expert: MAX 10 users
    - Each user: MAX 3 experts
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="expert_relations",
    )
    expert = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="user_relations",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "expert")
        verbose_name = "User-Expert Relation"
        verbose_name_plural = "User-Expert Relations"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} ↔ {self.expert.username}"


# ============================================================================
# Conversation & Messages
# ============================================================================

class Conversation(models.Model):
    """Chat conversation between a user and an expert."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="user_conversations",
    )
    expert = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="expert_conversations",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "expert")
        ordering = ["-updated_at"]
        verbose_name = "Conversation"
        verbose_name_plural = "Conversations"

    def __str__(self):
        return f"Chat: {self.user.username} ↔ {self.expert.username}"

    @property
    def last_message(self):
        """Return the most recent message in the conversation."""
        return self.messages.order_by("-created_at").first()

    def unread_count_for(self, user):
        """Count unread messages for a specific user."""
        return self.messages.filter(is_read=False).exclude(sender=user).count()


class Message(models.Model):
    """Individual message in a conversation."""

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="sent_messages",
    )
    text = models.TextField(blank=True)
    image = models.ImageField(upload_to="messages/images/", null=True, blank=True)
    file = models.FileField(upload_to="messages/files/", null=True, blank=True)
    file_name = models.CharField(max_length=255, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Message"
        verbose_name_plural = "Messages"

    def __str__(self):
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"{self.sender.username}: {preview}"


class MessageReaction(models.Model):
    """Emoji reaction on a message."""

    REACTION_CHOICES = [
        ("👍", "👍"),
        ("❤️", "❤️"),
        ("😂", "😂"),
        ("😮", "😮"),
        ("😢", "😢"),
    ]

    message = models.ForeignKey(
        Message,
        on_delete=models.CASCADE,
        related_name="reactions",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="message_reactions",
    )
    emoji = models.CharField(max_length=10, choices=REACTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("message", "user", "emoji")
        verbose_name = "Message Reaction"
        verbose_name_plural = "Message Reactions"

    def __str__(self):
        return f"{self.user.username} → {self.emoji}"


# ============================================================================
# Expert Complaint (Separate from User Complaints)
# ============================================================================

class ExpertComplaint(models.Model):
    """Complaint submitted by an expert. Separate from user complaints."""

    expert = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="expert_complaints",
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
        verbose_name = "Expert Complaint"
        verbose_name_plural = "Expert Complaints"

    def __str__(self):
        return f"[{self.status}] {self.title} — {self.expert.username}"


# ============================================================================
# Admin ↔ Expert Messaging
# ============================================================================

class AdminConversation(models.Model):
    """Thread-based conversation between Admin and Expert."""

    admin = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="admin_conversations_as_admin",
    )
    expert = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="admin_conversations_as_expert",
    )
    subject = models.CharField(max_length=200, default="General")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("admin", "expert")
        ordering = ["-updated_at"]
        verbose_name = "Admin Conversation"
        verbose_name_plural = "Admin Conversations"

    def __str__(self):
        return f"Admin Chat: {self.admin.username} ↔ {self.expert.username}"

    @property
    def last_message(self):
        """Return the most recent message."""
        return self.admin_messages.order_by("-created_at").first()

    def unread_count_for(self, user):
        """Count unread messages for a specific user."""
        return self.admin_messages.filter(is_read=False).exclude(sender=user).count()


class AdminMessage(models.Model):
    """Individual message in an Admin ↔ Expert conversation."""

    conversation = models.ForeignKey(
        AdminConversation,
        on_delete=models.CASCADE,
        related_name="admin_messages",
    )
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="admin_sent_messages",
    )
    text = models.TextField(blank=True)
    image = models.ImageField(upload_to="admin_messages/", null=True, blank=True)
    reactions = models.JSONField(default=dict, blank=True, help_text="Emoji reactions: {emoji: [user_ids]}")
    is_read = models.BooleanField(default=False)
    is_deleted = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Admin Message"
        verbose_name_plural = "Admin Messages"

    def __str__(self):
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"{self.sender.username}: {preview}"
