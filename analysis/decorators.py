# ============================================================================
# analysis/decorators.py
# Subscription Enforcement Decorators
# ============================================================================

from functools import wraps

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect

from .services.subscription import check_chat_access, check_scan_limit


def scan_limit_required(view_func):
    """
    Decorator that checks the user's daily scan limit before allowing
    access to scan endpoints.

    For AJAX/JSON requests: returns 403 JSON response.
    For regular requests: redirects to home with a warning message.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")

        result = check_scan_limit(request.user)

        if not result["allowed"]:
            # JSON/AJAX request
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or \
               request.content_type == "application/json":
                return JsonResponse({
                    "error": result["message"],
                    "limit_reached": True,
                    "plan_type": result["plan_type"],
                    "limit": result["limit"],
                }, status=403)

            messages.warning(request, result["message"])
            return redirect("home")

        return view_func(request, *args, **kwargs)

    return wrapper


def premium_required(view_func):
    """
    Decorator that restricts access to Premium plan users only.
    Non-premium users receive a 403 with an upgrade message.

    For AJAX/JSON requests: returns 403 JSON response.
    For regular requests: redirects to plans page with a message.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")

        allowed, message = check_chat_access(request.user)

        if not allowed:
            # JSON/AJAX request
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or \
               request.content_type == "application/json":
                return JsonResponse({
                    "error": message,
                    "upgrade_required": True,
                    "plan_type": request.user.plan_type,
                }, status=403)

            messages.warning(request, message)
            return redirect("plans")

        return view_func(request, *args, **kwargs)

    return wrapper
