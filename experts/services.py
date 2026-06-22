# ============================================================================
# experts/services.py
# Expert System — Business Logic
# ============================================================================

import logging
from typing import Optional, Tuple

from django.conf import settings
from django.db import IntegrityError
from django.utils import timezone

from analysis.models import CustomUser

logger = logging.getLogger(__name__)

# ── Relationship Limits ──────────────────────────────────────────────────────
MAX_USERS_PER_EXPERT = 10
MAX_EXPERTS_PER_USER = 3


# ============================================================================
# Expert Account Management
# ============================================================================

def create_expert_account(
    admin_user: CustomUser,
    username: str,
    email: str,
    password: str,
    specialization: str,
    first_name: str = "",
    last_name: str = "",
    bio: str = "",
) -> Tuple[bool, str, Optional[CustomUser]]:
    """
    Create an expert account. Only callable by admin/staff.

    Returns:
        (success, message, user_or_none)
    """
    if not admin_user.is_staff:
        return False, "Only administrators can create expert accounts.", None

    # Validate
    if CustomUser.objects.filter(username=username).exists():
        return False, f"Username '{username}' is already taken.", None

    if CustomUser.objects.filter(email=email).exists():
        return False, f"Email '{email}' is already in use.", None

    if not specialization.strip():
        return False, "Specialization is required.", None

    try:
        from experts.models import ExpertProfile

        # Create user with expert role
        user = CustomUser.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            role="expert",
            plan_type="premium",  # Experts get premium access
            email_verified=True,  # Admin-created accounts are pre-verified
        )

        # Create expert profile
        ExpertProfile.objects.create(
            user=user,
            specialization=specialization,
            bio=bio,
            is_active=True,
        )

        # Send verification email to expert
        try:
            from analysis.services.email_service import send_verification_email
            send_verification_email(user)
        except Exception as e:
            logger.warning("Could not send verification email to expert %s: %s", username, e)

        # Auto-assign users to this new expert
        try:
            from experts.auto_linking import auto_assign_users_to_expert
            auto_assign_users_to_expert(user)
        except Exception as e:
            logger.warning("Auto-linking failed for expert %s: %s", username, e)

        logger.info(
            "EXPERT CREATED: %s (specialization=%s) by admin=%s",
            username, specialization, admin_user.username,
        )
        return True, f"Expert account '{username}' created successfully.", user

    except Exception as e:
        logger.error("Failed to create expert: %s", e)
        return False, f"Error creating expert: {str(e)}", None


# ============================================================================
# User-Expert Relationship Management
# ============================================================================

def assign_expert_to_user(
    admin_user: CustomUser,
    user: CustomUser,
    expert: CustomUser,
) -> Tuple[bool, str]:
    """
    Admin assigns an expert to a user. Enforces relationship limits.

    Limits:
    - Each expert: MAX 10 users
    - Each user: MAX 3 experts
    """
    from experts.models import UserExpertRelation, ExpertProfile

    if not admin_user.is_staff:
        return False, "Only administrators can assign experts."

    # Verify expert role
    if not expert.is_expert:
        return False, f"User '{expert.username}' is not an expert."

    # Check expert profile exists and is active
    try:
        profile = expert.expert_profile
        if not profile.is_active:
            return False, f"Expert '{expert.username}' is currently inactive."
    except ExpertProfile.DoesNotExist:
        return False, f"Expert profile not found for '{expert.username}'."

    # Check if relation already exists
    if UserExpertRelation.objects.filter(user=user, expert=expert).exists():
        return False, f"{user.username} is already assigned to expert {expert.username}."

    # Enforce expert limit (max users per expert)
    expert_user_count = UserExpertRelation.objects.filter(expert=expert).count()
    if expert_user_count >= MAX_USERS_PER_EXPERT:
        return False, (
            f"Expert '{expert.username}' has reached the maximum of "
            f"{MAX_USERS_PER_EXPERT} assigned users."
        )

    # Enforce user limit (max experts per user)
    user_expert_count = UserExpertRelation.objects.filter(user=user).count()
    if user_expert_count >= MAX_EXPERTS_PER_USER:
        return False, (
            f"User '{user.username}' has reached the maximum of "
            f"{MAX_EXPERTS_PER_USER} assigned experts."
        )

    try:
        UserExpertRelation.objects.create(user=user, expert=expert)
        logger.info(
            "EXPERT ASSIGNED: user=%s expert=%s by admin=%s",
            user.username, expert.username, admin_user.username,
        )
        return True, f"Expert '{expert.username}' assigned to '{user.username}' successfully."

    except IntegrityError:
        return False, "This assignment already exists."


def remove_expert_from_user(
    admin_user: CustomUser,
    user: CustomUser,
    expert: CustomUser,
) -> Tuple[bool, str]:
    """Admin removes an expert from a user."""
    from experts.models import UserExpertRelation

    if not admin_user.is_staff:
        return False, "Only administrators can remove expert assignments."

    deleted, _ = UserExpertRelation.objects.filter(user=user, expert=expert).delete()

    if deleted:
        logger.info(
            "EXPERT REMOVED: user=%s expert=%s by admin=%s",
            user.username, expert.username, admin_user.username,
        )
        return True, f"Expert '{expert.username}' removed from '{user.username}'."
    return False, "Assignment not found."


# ============================================================================
# Messaging
# ============================================================================

def get_or_create_conversation(user: CustomUser, expert: CustomUser):
    """Get existing or create new conversation between user and expert."""
    from experts.models import Conversation

    conv, created = Conversation.objects.get_or_create(
        user=user,
        expert=expert,
    )
    if created:
        logger.info("CONVERSATION CREATED: user=%s expert=%s", user.username, expert.username)
    return conv


def send_message(sender: CustomUser, conversation_id: int, text: str, image=None, file=None):
    """
    Send a message in a conversation.

    Args:
        sender: User sending the message
        conversation_id: ID of the conversation
        text: Message text (can be empty if image/file provided)
        image: Optional uploaded image file
        file: Optional uploaded file

    Returns:
        (success, message_obj_or_error_string)
    """
    from experts.models import Conversation, Message

    try:
        conversation = Conversation.objects.get(pk=conversation_id)
    except Conversation.DoesNotExist:
        return False, "Conversation not found."

    # Verify sender is part of this conversation
    if sender != conversation.user and sender != conversation.expert:
        return False, "You are not part of this conversation."

    has_content = bool(text.strip()) or image or file
    if not has_content:
        return False, "Message cannot be empty."

    if text and len(text) > 5000:
        return False, "Message too long (max 5000 characters)."

    msg = Message(
        conversation=conversation,
        sender=sender,
        text=text.strip() if text else "",
    )

    if image:
        msg.image = image
    if file:
        msg.file = file
        msg.file_name = file.name if hasattr(file, 'name') else ""

    msg.save()

    # Update conversation timestamp
    conversation.updated_at = timezone.now()
    conversation.save(update_fields=["updated_at"])

    return True, msg


def mark_messages_read(user: CustomUser, conversation_id: int):
    """Mark all unread messages in a conversation as read for this user."""
    from experts.models import Conversation, Message

    try:
        conversation = Conversation.objects.get(pk=conversation_id)
    except Conversation.DoesNotExist:
        return

    # Mark messages from the other party as read
    Message.objects.filter(
        conversation=conversation,
        is_read=False,
    ).exclude(sender=user).update(is_read=True)


def get_unread_count(user: CustomUser) -> int:
    """Count total unread messages for a user across all conversations."""
    from experts.models import Message, Conversation

    # Get all conversations this user is part of
    from django.db.models import Q
    conversations = Conversation.objects.filter(Q(user=user) | Q(expert=user))

    return Message.objects.filter(
        conversation__in=conversations,
        is_read=False,
    ).exclude(sender=user).count()


def get_user_experts(user: CustomUser):
    """Get all experts assigned to a user with their profiles."""
    from experts.models import UserExpertRelation

    relations = (
        UserExpertRelation.objects
        .filter(user=user)
        .select_related("expert", "expert__expert_profile")
    )
    return relations


def get_expert_users(expert: CustomUser):
    """Get all users assigned to an expert."""
    from experts.models import UserExpertRelation

    relations = (
        UserExpertRelation.objects
        .filter(expert=expert)
        .select_related("user")
    )
    return relations
