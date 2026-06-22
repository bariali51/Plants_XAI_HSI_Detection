# ============================================================================
# community/forms.py
# Agricultural Social Platform — Community Forms
# ============================================================================

from django import forms
from .models import Post, Comment, PostReport


class PostForm(forms.ModelForm):
    """Form for creating and editing community posts."""

    tag_names = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            "placeholder": "e.g. tomato, blight, irrigation",
            "class": "tag-input",
        }),
        help_text="Comma-separated tags",
    )

    class Meta:
        model = Post
        fields = ["title", "body", "post_type"]
        widgets = {
            "title": forms.TextInput(attrs={
                "placeholder": "What's your agricultural question or topic?",
                "class": "form-input",
                "maxlength": 300,
            }),
            "body": forms.Textarea(attrs={
                "placeholder": "Describe your problem, question, or share your solution...",
                "class": "form-textarea",
                "rows": 6,
            }),
            "post_type": forms.Select(attrs={
                "class": "form-select",
            }),
        }

    def clean_title(self):
        title = self.cleaned_data["title"].strip()
        if len(title) < 5:
            raise forms.ValidationError("Title must be at least 5 characters.")
        return title

    def clean_body(self):
        body = self.cleaned_data["body"].strip()
        if len(body) < 10:
            raise forms.ValidationError("Post body must be at least 10 characters.")
        return body


class CommentForm(forms.ModelForm):
    """Form for adding comments to posts."""

    class Meta:
        model = Comment
        fields = ["body"]
        widgets = {
            "body": forms.Textarea(attrs={
                "placeholder": "Write your comment...",
                "class": "form-textarea comment-input",
                "rows": 3,
            }),
        }

    def clean_body(self):
        body = self.cleaned_data["body"].strip()
        if len(body) < 2:
            raise forms.ValidationError("Comment must be at least 2 characters.")
        if len(body) > 5000:
            raise forms.ValidationError("Comment is too long (max 5000 characters).")
        return body


class PostReportForm(forms.ModelForm):
    """Form for reporting a post."""

    class Meta:
        model = PostReport
        fields = ["reason", "detail"]
        widgets = {
            "reason": forms.Select(attrs={"class": "form-select"}),
            "detail": forms.Textarea(attrs={
                "placeholder": "Provide additional details (optional)...",
                "class": "form-textarea",
                "rows": 3,
            }),
        }
