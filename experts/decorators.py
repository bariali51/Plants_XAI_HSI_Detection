# ============================================================================
# experts/decorators.py
# Expert System — Access Control Decorators
# ============================================================================

from functools import wraps

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect


def expert_required(view_func):
    """
    Decorator that restricts access to expert role users only.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")

        if not request.user.is_expert:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse({"error": "Expert access required."}, status=403)
            messages.error(request, "You do not have expert access.")
            return redirect("home")

        return view_func(request, *args, **kwargs)

    return wrapper


def expert_or_staff_required(view_func):
    """
    Decorator that restricts access to expert or staff users.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")

        if not (request.user.is_expert or request.user.is_staff):
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse({"error": "Expert or admin access required."}, status=403)
            messages.error(request, "You do not have permission to access this area.")
            return redirect("home")

        return view_func(request, *args, **kwargs)

    return wrapper


def conversation_participant_required(view_func):
    """
    Decorator that ensures the user is a participant in the conversation.
    Expects 'conversation_id' in URL kwargs.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")

        from experts.models import Conversation

        conversation_id = kwargs.get("conversation_id")
        if conversation_id:
            try:
                conv = Conversation.objects.get(pk=conversation_id)
                if request.user != conv.user and request.user != conv.expert:
                    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                        return JsonResponse({"error": "Not authorized."}, status=403)
                    messages.error(request, "You are not part of this conversation.")
                    return redirect("home")
            except Conversation.DoesNotExist:
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return JsonResponse({"error": "Conversation not found."}, status=404)
                messages.error(request, "Conversation not found.")
                return redirect("home")

        return view_func(request, *args, **kwargs)

    return wrapper
