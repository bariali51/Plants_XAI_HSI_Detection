# ============================================================================
# analysis/services/email_service.py
# Email Verification Service — 6-Digit Code Based
# ============================================================================

import logging
import random

from django.conf import settings
from django.core.mail import send_mail
from django.utils.html import strip_tags

logger = logging.getLogger(__name__)


def generate_verification_code():
    """Generate a random 6-digit verification code."""
    return str(random.randint(100000, 999999))


def send_verification_email(user, request=None):
    """
    Generate a 6-digit code, store it, and send it via email.

    Args:
        user: CustomUser instance
        request: HttpRequest (unused, kept for API compatibility)

    Returns:
        (success, message)
    """
    from analysis.models import EmailVerificationCode

    # Invalidate any existing unused codes for this user
    EmailVerificationCode.objects.filter(user=user, is_used=False).update(is_used=True)

    # Generate new code
    code = generate_verification_code()
    EmailVerificationCode.objects.create(user=user, code=code)

    subject = "Nabtati — Your Verification Code"

    html_message = f"""
    <html>
    <body style="font-family: 'Inter', Arial, sans-serif; background: #f8fafc; padding: 40px 20px;">
        <div style="max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 16px;
                    box-shadow: 0 4px 24px rgba(0,0,0,0.08); overflow: hidden;">
            <div style="background: linear-gradient(135deg, #2e7d32, #1b5e20); padding: 32px; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 24px;">🌿 Nabtati</h1>
                <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 14px;">
                    Email Verification
                </p>
            </div>
            <div style="padding: 32px;">
                <h2 style="color: #1e293b; font-size: 20px; margin: 0 0 12px;">
                    Welcome, {user.first_name or user.username}!
                </h2>
                <p style="color: #64748b; line-height: 1.6; font-size: 14px;">
                    Your verification code is:
                </p>
                <div style="text-align: center; margin: 24px 0;">
                    <div style="display: inline-block; padding: 16px 40px;
                                background: linear-gradient(135deg, #f0fdf4, #dcfce7);
                                border: 2px solid #86efac; border-radius: 16px;">
                        <span style="font-size: 36px; font-weight: 900; letter-spacing: 8px;
                                     color: #166534; font-family: monospace;">{code}</span>
                    </div>
                </div>
                <p style="color: #64748b; line-height: 1.6; font-size: 14px;">
                    Enter this code on the verification page to activate your account.
                </p>
                <p style="color: #94a3b8; font-size: 12px; line-height: 1.6; margin-top: 16px;">
                    This code will expire in <strong>15 minutes</strong>. If you didn't create
                    this account, you can safely ignore this email.
                </p>
            </div>
            <div style="background: #f8fafc; padding: 16px 32px; text-align: center; border-top: 1px solid #e2e8f0;">
                <p style="color: #94a3b8; font-size: 11px; margin: 0;">
                    Nabtati — Smart Plant Disease Detection Platform
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    plain_message = strip_tags(html_message)

    try:
        send_mail(
            subject=subject,
            message=plain_message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[user.email],
            html_message=html_message,
            fail_silently=False,
        )
        logger.info("Verification code sent to %s (user=%s)", user.email, user.username)
        return True, "Verification code sent successfully."
    except Exception as e:
        logger.error("Failed to send verification email to %s: %s", user.email, e)
        return False, f"Failed to send verification email: {str(e)}"


def verify_code(user_pk, code):
    """
    Verify a 6-digit email verification code.

    Args:
        user_pk: Primary key of the user
        code: The 6-digit code entered by user

    Returns:
        (success, user_or_error_message)
    """
    from analysis.models import EmailVerificationCode, CustomUser

    try:
        user = CustomUser.objects.get(pk=user_pk)
    except CustomUser.DoesNotExist:
        return False, "User not found."

    # Find matching code
    try:
        verification = EmailVerificationCode.objects.filter(
            user=user,
            code=code,
            is_used=False,
        ).latest("created_at")
    except EmailVerificationCode.DoesNotExist:
        return False, "Invalid verification code. Please check and try again."

    # Check expiry
    if verification.is_expired():
        return False, "This code has expired. Please request a new one."

    # Mark code as used
    verification.is_used = True
    verification.save(update_fields=["is_used"])

    # Activate user
    user.email_verified = True
    user.save(update_fields=["email_verified"])

    logger.info("Email verified for user %s", user.username)
    return True, user


# Legacy compatibility — keep verify_token as wrapper
def verify_token(token, max_age=86400):
    """Legacy token verification — redirects to code-based system."""
    return False, "Please use the 6-digit code verification instead."
