# ============================================================================
# community/templatetags/community_tags.py
# Custom template tags for the community app
# ============================================================================

from django import template
from django.utils.timesince import timesince

register = template.Library()


@register.filter
def time_ago(value):
    """Render a datetime as 'X time ago' string."""
    if not value:
        return ""
    return timesince(value) + " ago"


@register.filter
def tag_color(category):
    """Return CSS class for tag category color."""
    colors = {
        "crop": "tag-crop",
        "disease": "tag-disease",
        "topic": "tag-topic",
        "service": "tag-service",
    }
    return colors.get(category, "tag-default")


@register.filter
def post_type_icon(post_type):
    """Return icon class for post type."""
    icons = {
        "question": "❓",
        "solution": "✅",
        "discussion": "💬",
        "alert": "⚠️",
    }
    return icons.get(post_type, "📝")


@register.filter
def post_type_class(post_type):
    """Return CSS class for post type badge."""
    classes = {
        "question": "badge-question",
        "solution": "badge-solution",
        "discussion": "badge-discussion",
        "alert": "badge-alert",
    }
    return classes.get(post_type, "badge-default")


@register.filter
def status_class(status):
    """Return CSS class for post status badge."""
    classes = {
        "active": "status-active",
        "resolved": "status-resolved",
        "closed": "status-closed",
    }
    return classes.get(status, "status-default")


@register.filter
def truncate_body(text, length=200):
    """Truncate text to specified length with ellipsis."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."


@register.filter
def multiply(value, arg):
    """Multiply the value by the argument."""
    try:
        return int(value) * int(arg)
    except (ValueError, TypeError):
        return value
