# ============================================================================
# experts/views.py
# Expert System — Views
# ============================================================================

import json

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import models
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from analysis.decorators import premium_required
from analysis.models import CustomUser, ScanResult

from .decorators import (
    conversation_participant_required,
    expert_required,
)
from .models import (
    Conversation,
    ExpertComplaint,
    ExpertProfile,
    Message,
    UserExpertRelation,
)
from .services import (
    get_expert_users,
    get_or_create_conversation,
    get_unread_count,
    get_user_experts,
    mark_messages_read,
    send_message,
)


# ============================================================================
# Expert Dashboard
# ============================================================================

@login_required
@expert_required
def expert_dashboard(request):
    """Expert home dashboard with key stats."""
    expert = request.user

    # Stats
    relations = get_expert_users(expert)
    user_count = relations.count()

    conversations = Conversation.objects.filter(expert=expert)
    unread = get_unread_count(expert)

    total_messages = Message.objects.filter(conversation__expert=expert).count()

    # Recent conversations
    recent_convs = conversations.select_related("user")[:5]

    context = {
        "user_count": user_count,
        "max_users": 10,
        "unread_messages": unread,
        "total_messages": total_messages,
        "recent_conversations": recent_convs,
        "relations": relations,
    }
    return render(request, "experts/expert_dashboard.html", context)


# ============================================================================
# Expert Messages
# ============================================================================

@login_required
@expert_required
def expert_messages(request):
    """List all conversations for the expert."""
    conversations = (
        Conversation.objects
        .filter(expert=request.user)
        .select_related("user")
        .order_by("-updated_at")
    )

    conv_data = []
    for conv in conversations:
        conv_data.append({
            "conversation": conv,
            "unread": conv.unread_count_for(request.user),
            "last_msg": conv.last_message,
        })

    context = {
        "conversations": conv_data,
        "total_unread": get_unread_count(request.user),
    }
    return render(request, "experts/expert_messages.html", context)


@login_required
@expert_required
@conversation_participant_required
def expert_conversation(request, conversation_id):
    """View a specific conversation with a user."""
    conversation = get_object_or_404(Conversation, pk=conversation_id, expert=request.user)

    # Mark messages as read
    mark_messages_read(request.user, conversation_id)

    messages_qs = conversation.messages.select_related("sender").order_by("created_at")

    # Get user's recent scans for context
    user_scans = ScanResult.objects.filter(user=conversation.user).order_by("-created_at")[:5]

    context = {
        "conversation": conversation,
        "messages_list": messages_qs,
        "other_user": conversation.user,
        "user_scans": user_scans,
    }
    return render(request, "experts/expert_conversation.html", context)


# ============================================================================
# Expert My Files (Expert's own scans)
# ============================================================================

@login_required
@expert_required
def expert_my_files(request):
    """Expert's own scan files."""
    scans = ScanResult.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "experts/expert_my_files.html", {"scans": scans})


# ============================================================================
# Expert Complaints
# ============================================================================

@login_required
@expert_required
def expert_complaints(request):
    """Expert complaint submission and history."""
    errors = {}

    if request.method == "POST":
        title = request.POST.get("title", "").strip()
        message_text = request.POST.get("message", "").strip()

        if not title:
            errors["title"] = "Title is required."
        elif len(title) < 3:
            errors["title"] = "Title must be at least 3 characters."

        if not message_text:
            errors["message"] = "Message is required."
        elif len(message_text) < 10:
            errors["message"] = "Message must be at least 10 characters."

        if not errors:
            ExpertComplaint.objects.create(
                expert=request.user,
                title=title,
                message=message_text,
            )
            messages.success(request, "Your complaint has been submitted successfully.")
            return redirect("experts:complaints")

    complaints = ExpertComplaint.objects.filter(expert=request.user)
    context = {"complaints": complaints, "errors": errors}
    return render(request, "experts/expert_complaints.html", context)


@login_required
@expert_required
@require_POST
def expert_delete_complaint(request, complaint_id):
    """Delete an expert's own complaint."""
    complaint = get_object_or_404(ExpertComplaint, pk=complaint_id, expert=request.user)
    complaint.delete()
    messages.success(request, "Complaint deleted successfully.")
    return redirect("experts:complaints")


# ============================================================================
# Expert Settings
# ============================================================================

ALLOWED_AVATAR_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2 MB


@login_required
@expert_required
def expert_settings(request):
    """Expert profile settings."""
    import os

    user = request.user
    profile = user.expert_profile
    errors = {}

    if request.method == "POST":
        action = request.POST.get("action", "")

        if action == "update_profile":
            first_name = request.POST.get("first_name", "").strip()
            last_name = request.POST.get("last_name", "").strip()
            specialization = request.POST.get("specialization", "").strip()
            title = request.POST.get("title", "").strip()
            bio = request.POST.get("bio", "").strip()
            skills_raw = request.POST.get("skills", "").strip()
            linkedin_url = request.POST.get("linkedin_url", "").strip()
            website_url = request.POST.get("website_url", "").strip()

            if not specialization:
                errors["specialization"] = "Specialization is required."

            # Parse skills: comma-separated → list, strip whitespace, remove empties
            skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

            if not errors:
                user.first_name = first_name
                user.last_name = last_name
                user.save(update_fields=["first_name", "last_name"])

                profile.specialization = specialization
                profile.title = title
                profile.bio = bio
                profile.skills = skills_list
                profile.linkedin_url = linkedin_url
                profile.website_url = website_url
                profile.save(update_fields=[
                    "specialization", "title", "bio",
                    "skills", "linkedin_url", "website_url",
                ])

                messages.success(request, "Profile updated successfully.")

        elif action == "update_avatar":
            avatar_file = request.FILES.get("avatar")
            if not avatar_file:
                errors["avatar"] = "Please select an image file."
            else:
                ext = os.path.splitext(avatar_file.name)[1].lower()
                if ext not in ALLOWED_AVATAR_EXTS:
                    errors["avatar"] = f"Image type '{ext}' is not allowed. Use JPG, PNG, GIF, or WebP."
                elif avatar_file.size > MAX_AVATAR_SIZE:
                    errors["avatar"] = "Image is too large (max 2 MB)."

            if not errors:
                # Delete old avatar file if it exists
                if profile.avatar:
                    profile.avatar.delete(save=False)
                profile.avatar = avatar_file
                profile.save(update_fields=["avatar"])
                messages.success(request, "Avatar updated successfully.")

        elif action == "remove_avatar":
            if profile.avatar:
                profile.avatar.delete(save=False)
                profile.avatar = None
                profile.save(update_fields=["avatar"])
                messages.success(request, "Avatar removed.")

        elif action == "change_password":
            current = request.POST.get("current_password", "")
            new_pw = request.POST.get("new_password", "")
            confirm = request.POST.get("confirm_password", "")

            if not user.check_password(current):
                errors["current_password"] = "Current password is incorrect."
            elif len(new_pw) < 6:
                errors["new_password"] = "Password must be at least 6 characters."
            elif new_pw != confirm:
                errors["confirm_password"] = "Passwords do not match."

            if not errors:
                user.set_password(new_pw)
                user.save()
                from django.contrib.auth import update_session_auth_hash
                update_session_auth_hash(request, user)
                messages.success(request, "Password changed successfully.")

    context = {"profile": profile, "errors": errors}
    return render(request, "experts/expert_settings.html", context)


# ============================================================================
# User-Facing: My Experts
# ============================================================================

@login_required
@premium_required
def user_experts(request):
    """Show user's assigned experts."""
    relations = get_user_experts(request.user)

    expert_data = []
    for rel in relations:
        expert = rel.expert
        try:
            profile = expert.expert_profile
        except ExpertProfile.DoesNotExist:
            continue

        # Get or create conversation
        conv = get_or_create_conversation(request.user, expert)
        unread = conv.unread_count_for(request.user)

        expert_data.append({
            "expert": expert,
            "profile": profile,
            "conversation": conv,
            "unread": unread,
        })

    context = {"experts": expert_data}
    return render(request, "experts/user_experts.html", context)


@login_required
@premium_required
@conversation_participant_required
def user_conversation(request, conversation_id):
    """User's chat view with an expert."""
    conversation = get_object_or_404(
        Conversation, pk=conversation_id, user=request.user
    )

    # Mark messages as read
    mark_messages_read(request.user, conversation_id)

    messages_qs = conversation.messages.select_related("sender").order_by("created_at")

    # User's scans for attaching context
    user_scans = ScanResult.objects.filter(user=request.user).order_by("-created_at")[:10]

    context = {
        "conversation": conversation,
        "messages_list": messages_qs,
        "other_user": conversation.expert,
        "user_scans": user_scans,
    }
    return render(request, "experts/user_conversation.html", context)


# ============================================================================
# API Endpoints (AJAX)
# ============================================================================

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ALLOWED_FILE_EXTS = {".pdf", ".doc", ".docx", ".txt", ".csv", ".xls", ".xlsx"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB


@login_required
@require_POST
def api_send_message(request):
    """POST — Send a message in a conversation (supports text + file/image)."""
    # Support both JSON and multipart form data
    content_type = request.content_type or ""

    if "multipart/form-data" in content_type:
        conversation_id = request.POST.get("conversation_id")
        text = request.POST.get("text", "").strip()
        uploaded_image = request.FILES.get("image")
        uploaded_file = request.FILES.get("file")
    else:
        try:
            data = json.loads(request.body)
            conversation_id = data.get("conversation_id")
            text = data.get("text", "").strip()
            uploaded_image = None
            uploaded_file = None
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)

    if not conversation_id:
        return JsonResponse({"error": "Conversation ID required."}, status=400)

    if not text and not uploaded_image and not uploaded_file:
        return JsonResponse({"error": "Message cannot be empty."}, status=400)

    # Validate uploads
    import os
    if uploaded_image:
        ext = os.path.splitext(uploaded_image.name)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTS:
            return JsonResponse({"error": f"Image type '{ext}' not allowed."}, status=400)
        if uploaded_image.size > MAX_UPLOAD_SIZE:
            return JsonResponse({"error": "Image too large (max 5MB)."}, status=400)

    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in ALLOWED_FILE_EXTS:
            return JsonResponse({"error": f"File type '{ext}' not allowed."}, status=400)
        if uploaded_file.size > MAX_UPLOAD_SIZE:
            return JsonResponse({"error": "File too large (max 5MB)."}, status=400)

    try:
        success, result = send_message(
            request.user,
            conversation_id,
            text,
            image=uploaded_image,
            file=uploaded_file,
        )

        if success:
            msg = result
            msg_data = {
                "id": msg.id,
                "text": msg.text,
                "sender": msg.sender.username,
                "is_mine": True,
                "created_at": msg.created_at.strftime("%H:%M"),
                "created_at_full": msg.created_at.strftime("%Y-%m-%d %H:%M"),
            }
            if msg.image:
                msg_data["image_url"] = msg.image.url
            if msg.file:
                msg_data["file_url"] = msg.file.url
                msg_data["file_name"] = msg.file_name or msg.file.name.split("/")[-1]
            return JsonResponse({"status": "ok", "message": msg_data})
        else:
            return JsonResponse({"error": result}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def api_get_messages(request, conversation_id):
    """GET — Poll for new messages in a conversation."""
    conversation = get_object_or_404(Conversation, pk=conversation_id)

    # Verify participant
    if request.user != conversation.user and request.user != conversation.expert:
        return JsonResponse({"error": "Not authorized."}, status=403)

    # Get messages after a certain ID
    after_id = request.GET.get("after", 0)
    try:
        after_id = int(after_id)
    except (ValueError, TypeError):
        after_id = 0

    new_messages = conversation.messages.filter(id__gt=after_id).select_related("sender")

    # Mark as read
    mark_messages_read(request.user, conversation_id)

    messages_data = []
    for msg in new_messages:
        entry = {
            "id": msg.id,
            "text": msg.text,
            "sender": msg.sender.username,
            "is_mine": msg.sender == request.user,
            "created_at": msg.created_at.strftime("%H:%M"),
            "created_at_full": msg.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        if msg.image:
            entry["image_url"] = msg.image.url
        if msg.file:
            entry["file_url"] = msg.file.url
            entry["file_name"] = msg.file_name or msg.file.name.split("/")[-1]

        # Include reactions
        from .models import MessageReaction
        reactions = MessageReaction.objects.filter(message=msg).values("emoji").annotate(
            count=models.Count("id")
        )
        entry["reactions"] = {r["emoji"]: r["count"] for r in reactions}
        # Check which reactions the current user has made
        my_reactions = list(
            MessageReaction.objects.filter(message=msg, user=request.user).values_list("emoji", flat=True)
        )
        entry["my_reactions"] = my_reactions

        messages_data.append(entry)

    return JsonResponse({"status": "ok", "messages": messages_data})


@login_required
@require_POST
def api_mark_read(request, conversation_id):
    """POST — Mark all messages in conversation as read."""
    mark_messages_read(request.user, conversation_id)
    return JsonResponse({"status": "ok"})


@login_required
@require_POST
def api_delete_message(request, message_id):
    """POST — Delete a message (only own messages)."""
    from .models import Message

    msg = get_object_or_404(Message, pk=message_id)

    # Only the sender can delete
    if msg.sender != request.user:
        return JsonResponse({"error": "Not authorized."}, status=403)

    # Verify user is a participant
    conv = msg.conversation
    if request.user != conv.user and request.user != conv.expert:
        return JsonResponse({"error": "Not authorized."}, status=403)

    msg.delete()
    return JsonResponse({"status": "ok"})


@login_required
@require_POST
def api_toggle_reaction(request, message_id):
    """POST — Toggle an emoji reaction on a message."""
    import json
    from .models import Message, MessageReaction

    msg = get_object_or_404(Message, pk=message_id)

    # Verify user is a participant
    conv = msg.conversation
    if request.user != conv.user and request.user != conv.expert:
        return JsonResponse({"error": "Not authorized."}, status=403)

    try:
        data = json.loads(request.body)
        emoji = data.get("emoji", "")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    valid_emojis = ["👍", "❤️", "😂", "😮", "😢"]
    if emoji not in valid_emojis:
        return JsonResponse({"error": "Invalid emoji."}, status=400)

    # Toggle: if exists → remove, else → add
    existing = MessageReaction.objects.filter(
        message=msg, user=request.user, emoji=emoji
    ).first()

    if existing:
        existing.delete()
        action = "removed"
    else:
        MessageReaction.objects.create(message=msg, user=request.user, emoji=emoji)
        action = "added"

    # Return updated reactions for this message
    reactions = MessageReaction.objects.filter(message=msg).values("emoji").annotate(
        count=models.Count("id")
    )
    reaction_data = {r["emoji"]: r["count"] for r in reactions}
    my_reactions = list(
        MessageReaction.objects.filter(message=msg, user=request.user).values_list("emoji", flat=True)
    )

    return JsonResponse({
        "status": "ok",
        "action": action,
        "reactions": reaction_data,
        "my_reactions": my_reactions,
    })


# ============================================================================
# Admin ↔ Expert Messaging (Expert Side)
# ============================================================================

@login_required
@expert_required
def expert_admin_messages(request):
    """Expert inbox for admin conversations."""
    from .models import AdminConversation

    conversations = (
        AdminConversation.objects
        .filter(expert=request.user)
        .select_related("admin")
        .order_by("-updated_at")
    )

    conv_data = []
    for conv in conversations:
        conv_data.append({
            "conversation": conv,
            "unread": conv.unread_count_for(request.user),
            "last_msg": conv.last_message,
        })

    total_unread = sum(c["unread"] for c in conv_data)

    context = {
        "conversations": conv_data,
        "total_unread": total_unread,
    }
    return render(request, "experts/expert_admin_messages.html", context)


@login_required
@expert_required
def expert_admin_conversation(request, conversation_id):
    """Expert view of a specific admin conversation."""
    from .models import AdminConversation, AdminMessage

    conversation = get_object_or_404(
        AdminConversation, pk=conversation_id, expert=request.user
    )

    # Mark messages as read
    AdminMessage.objects.filter(
        conversation=conversation, is_read=False
    ).exclude(sender=request.user).update(is_read=True)

    messages_qs = conversation.admin_messages.select_related("sender").order_by("created_at")

    context = {
        "conversation": conversation,
        "messages_list": messages_qs,
        "other_user": conversation.admin,
    }
    return render(request, "experts/expert_admin_conversation.html", context)


@login_required
@expert_required
@require_POST
def api_expert_admin_reply(request):
    """POST — Expert sends a reply in an admin conversation."""
    from .models import AdminConversation, AdminMessage

    # Support both FormData (with files) and JSON
    if request.content_type and "multipart" in request.content_type:
        conversation_id = request.POST.get("conversation_id")
        text = request.POST.get("text", "").strip()
        image = request.FILES.get("image")
    else:
        try:
            data = json.loads(request.body)
            conversation_id = data.get("conversation_id")
            text = data.get("text", "").strip()
            image = None
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not conversation_id or (not text and not image):
        return JsonResponse({"error": "Conversation ID and text or image required"}, status=400)

    conversation = get_object_or_404(
        AdminConversation, pk=conversation_id, expert=request.user
    )

    msg = AdminMessage.objects.create(
        conversation=conversation,
        sender=request.user,
        text=text,
        image=image,
    )

    conversation.save()  # Update updated_at

    return JsonResponse({
        "status": "ok",
        "message": {
            "id": msg.id,
            "text": msg.text,
            "image_url": msg.image.url if msg.image else None,
            "sender": msg.sender.username,
            "is_mine": True,
            "created_at": msg.created_at.strftime("%H:%M"),
        },
    })


@login_required
@expert_required
def api_expert_admin_poll(request, conversation_id):
    """GET — Poll for new messages in an admin conversation (expert side)."""
    from .models import AdminConversation, AdminMessage

    conversation = get_object_or_404(
        AdminConversation, pk=conversation_id, expert=request.user
    )

    after_id = request.GET.get("after", 0)
    try:
        after_id = int(after_id)
    except (ValueError, TypeError):
        after_id = 0

    new_messages = conversation.admin_messages.filter(
        id__gt=after_id
    ).select_related("sender")

    # Mark as read
    AdminMessage.objects.filter(
        conversation=conversation, is_read=False
    ).exclude(sender=request.user).update(is_read=True)

    messages_data = []
    user_id = str(request.user.id)
    for msg in new_messages:
        messages_data.append({
            "id": msg.id,
            "text": msg.text,
            "image_url": msg.image.url if msg.image else None,
            "sender": msg.sender.username,
            "is_mine": msg.sender == request.user,
            "is_deleted": msg.is_deleted,
            "reactions": {k: len(v) for k, v in (msg.reactions or {}).items()},
            "my_reactions": [k for k, v in (msg.reactions or {}).items() if user_id in v],
            "created_at": msg.created_at.strftime("%H:%M"),
        })

    return JsonResponse({"status": "ok", "messages": messages_data})


@login_required
@expert_required
@require_POST
def api_expert_admin_react(request, message_id):
    """POST — Toggle an emoji reaction on an admin message."""
    from .models import AdminMessage

    msg = get_object_or_404(AdminMessage, pk=message_id)
    if msg.conversation.expert != request.user:
        return JsonResponse({"error": "Forbidden"}, status=403)

    try:
        data = json.loads(request.body)
        emoji = data.get("emoji", "").strip()
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not emoji:
        return JsonResponse({"error": "Emoji required"}, status=400)

    reactions = msg.reactions or {}
    user_id = str(request.user.id)

    if emoji not in reactions:
        reactions[emoji] = []

    if user_id in reactions[emoji]:
        reactions[emoji].remove(user_id)
        action = "removed"
        if not reactions[emoji]:
            del reactions[emoji]
    else:
        reactions[emoji].append(user_id)
        action = "added"

    msg.reactions = reactions
    msg.save(update_fields=["reactions"])

    return JsonResponse({
        "status": "ok",
        "action": action,
        "reactions": {k: len(v) for k, v in msg.reactions.items()},
        "my_reactions": [k for k, v in msg.reactions.items() if user_id in v],
    })


@login_required
@expert_required
@require_POST
def api_expert_admin_delete_msg(request, message_id):
    """POST — Soft-delete a message (only sender can delete)."""
    from .models import AdminMessage

    msg = get_object_or_404(AdminMessage, pk=message_id)

    if msg.sender != request.user:
        return JsonResponse({"error": "You can only delete your own messages"}, status=403)

    msg.is_deleted = True
    msg.text = ""
    if msg.image:
        msg.image.delete(save=False)
        msg.image = None
    msg.save(update_fields=["is_deleted", "text", "image"])

    return JsonResponse({"status": "ok", "message_id": message_id})


# ============================================================================
# Public Expert Profile
# ============================================================================

@login_required
def public_expert_profile(request, user_id):
    """View an expert's public profile. Admins see full details; others see limited."""
    from .models import ExpertProfile

    expert_user = get_object_or_404(CustomUser, pk=user_id, role="expert")
    profile = get_object_or_404(ExpertProfile, user=expert_user)

    # Role-based access control
    is_admin = request.user.is_staff
    is_owner = request.user == expert_user

    context = {
        "expert_user": expert_user,
        "profile": profile,
        "is_admin": is_admin,
        "is_owner": is_owner,
        "show_sensitive": is_admin or is_owner,  # email, social links
    }
    return render(request, "experts/expert_public_profile.html", context)
