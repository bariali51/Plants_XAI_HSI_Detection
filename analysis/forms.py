from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser
import re

class SignUpForm(UserCreationForm):

    class Meta:
        model = CustomUser
        fields = [
            "first_name",
            "last_name",
            "birth_date", 
            "email",
            "username",
            "purpose",
            "password1",
            "password2",
        ]

    def clean_username(self):
        username = self.cleaned_data["username"]

        if len(username) < 8:
            raise forms.ValidationError("Username must be at least 8 characters")

        if not re.search(r"[A-Z]", username):
            raise forms.ValidationError("Must contain uppercase letter")

        if not re.search(r"[a-z]", username):
            raise forms.ValidationError("Must contain lowercase letter")

        if not re.search(r"[0-9]", username):
            raise forms.ValidationError("Must contain number")

        return username