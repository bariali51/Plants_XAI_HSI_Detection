# ============================================================================
# manager/decorators.py
# Staff-Only Access Control
# ============================================================================

from functools import wraps

from django.contrib import messages
from django.shortcuts import redirect


def staff_required(view_func):
    """
    Decorator that restricts access to staff users only.
    Non-authenticated users are redirected to login.
    Authenticated non-staff users are redirected to home with an error message.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.error(request, "Please log in to access this page.")
            return redirect("login")

        if not request.user.is_staff:
            messages.error(request, "You do not have permission to access this area.")
            return redirect("home")

        return view_func(request, *args, **kwargs)

    return wrapper
