# ============================================================================
# analysis/middleware.py
# Custom Middleware — Logging, Security Headers, Rate Limiting
# ============================================================================

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable

from django.http import HttpRequest, HttpResponse, JsonResponse

logger = logging.getLogger(__name__)


# ============================================================================
# Request Logging Middleware
# ============================================================================

class RequestLoggingMiddleware:
    """
    Logs every request with method, path, status code, and duration.

    Example log line:
        [INFO] POST /api/ai/chat/ -> 200 (142ms) user=admin
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        start = time.time()
        response = self.get_response(request)
        duration_ms = (time.time() - start) * 1000

        # Skip logging for static/media files to reduce noise
        path = request.path
        if not path.startswith(("/static/", "/media/", "/favicon")):
            user = getattr(request, "user", None)
            username = getattr(user, "username", "anonymous") if user else "anonymous"

            logger.info(
                "%s %s -> %d (%.0fms) user=%s",
                request.method,
                path,
                response.status_code,
                duration_ms,
                username,
            )

        return response


# ============================================================================
# Security Headers Middleware
# ============================================================================

class SecurityHeadersMiddleware:
    """
    Adds essential security headers to every response.
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)

        # Prevent MIME type sniffing
        response["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking (reinforced beyond Django's built-in)
        if "X-Frame-Options" not in response:
            response["X-Frame-Options"] = "DENY"

        # Referrer policy
        response["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy (limit browser features)
        response["Permissions-Policy"] = (
            "camera=(self), microphone=(), geolocation=(), payment=()"
        )

        return response


# ============================================================================
# Upload Rate Limiting Middleware
# ============================================================================

class UploadRateLimitMiddleware:
    """
    Rate-limits file upload endpoints to prevent abuse.

    Limits: 15 uploads per minute per IP address.
    Only applies to POST requests with files attached.
    """

    RATE_LIMIT = 15  # max uploads per window
    WINDOW_SECONDS = 60  # time window

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response
        self._requests: dict = defaultdict(list)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Only rate-limit POST requests with file uploads
        if request.method == "POST" and request.FILES:
            ip = self._get_client_ip(request)
            now = time.time()

            # Clean old entries
            self._requests[ip] = [
                t for t in self._requests[ip]
                if now - t < self.WINDOW_SECONDS
            ]

            if len(self._requests[ip]) >= self.RATE_LIMIT:
                logger.warning(
                    "Rate limit exceeded for IP %s (%d uploads in %ds)",
                    ip, len(self._requests[ip]), self.WINDOW_SECONDS,
                )
                return JsonResponse(
                    {
                        "error": "Too many uploads. Please wait a minute before trying again.",
                    },
                    status=429,
                )

            self._requests[ip].append(now)

        return self.get_response(request)

    @staticmethod
    def _get_client_ip(request: HttpRequest) -> str:
        """Extract client IP, respecting X-Forwarded-For behind proxies."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "0.0.0.0")


# ============================================================================
# Subscription Validation Middleware
# ============================================================================

class SubscriptionMiddleware:
    """
    Auto-validates and expires subscriptions on every request.

    - Checks if the authenticated user's subscription has expired.
    - If expired, downgrades to free plan automatically.
    - Lightweight: only writes to DB when an actual expiry occurs.
    """

    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        user = getattr(request, "user", None)

        if user and getattr(user, "is_authenticated", False):
            # Skip for free plan users (nothing to expire)
            plan = getattr(user, "plan_type", "free")
            if plan != "free":
                expired = user.check_and_expire_subscription()
                if expired:
                    logger.info(
                        "SUBSCRIPTION EXPIRED: user=%s downgraded to free",
                        user.username,
                    )

        return self.get_response(request)
