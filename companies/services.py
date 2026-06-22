# ============================================================================
# companies/services.py
# Agricultural Social Platform — Company Messaging Business Logic
# ============================================================================

import logging
from django.utils import timezone

logger = logging.getLogger(__name__)


def get_or_create_company_conversation(user, company_profile):
    """Get existing or create new conversation between user and company."""
    from .models import CompanyConversation

    conv, created = CompanyConversation.objects.get_or_create(
        user=user,
        company=company_profile,
    )
    if created:
        logger.info(
            "COMPANY CONVERSATION CREATED: user=%s company=%s",
            user.username, company_profile.company_name,
        )
    return conv


def send_company_message(sender, conversation_id, text, image=None, file=None):
    """
    Send a message in a user↔company conversation.

    Returns:
        (success, message_obj_or_error_string)
    """
    from .models import CompanyConversation, CompanyMessage

    try:
        conversation = CompanyConversation.objects.get(pk=conversation_id)
    except CompanyConversation.DoesNotExist:
        return False, "Conversation not found."

    # Verify sender is part of this conversation
    if sender != conversation.user and sender != conversation.company.user:
        return False, "You are not part of this conversation."

    text = (text or "").strip()
    has_content = bool(text) or image or file
    if not has_content:
        return False, "Message cannot be empty."

    if text and len(text) > 5000:
        return False, "Message too long (max 5000 characters)."

    msg = CompanyMessage(
        conversation=conversation,
        sender=sender,
        text=text,
    )

    if image:
        msg.image = image
    if file:
        msg.file = file
        msg.file_name = file.name if hasattr(file, "name") else ""

    msg.save()

    # Update conversation timestamp
    conversation.updated_at = timezone.now()
    conversation.save(update_fields=["updated_at"])

    return True, msg


def mark_company_messages_read(user, conversation_id):
    """Mark all unread messages in a conversation as read for this user."""
    from .models import CompanyConversation, CompanyMessage

    try:
        conversation = CompanyConversation.objects.get(pk=conversation_id)
    except CompanyConversation.DoesNotExist:
        return

    CompanyMessage.objects.filter(
        conversation=conversation,
        is_read=False,
    ).exclude(sender=user).update(is_read=True)


def get_company_unread_count(user):
    """Count total unread messages for a company user across all conversations."""
    from .models import CompanyMessage, CompanyConversation, CompanyProfile

    try:
        profile = user.company_profile
    except CompanyProfile.DoesNotExist:
        return 0

    conversations = CompanyConversation.objects.filter(company=profile)
    return CompanyMessage.objects.filter(
        conversation__in=conversations,
        is_read=False,
    ).exclude(sender=user).count()


def get_user_company_unread_count(user):
    """Count total unread messages for a regular user across company conversations."""
    from .models import CompanyMessage, CompanyConversation

    conversations = CompanyConversation.objects.filter(user=user)
    return CompanyMessage.objects.filter(
        conversation__in=conversations,
        is_read=False,
    ).exclude(sender=user).count()
