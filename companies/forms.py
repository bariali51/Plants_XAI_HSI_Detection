# ============================================================================
# companies/forms.py
# Agricultural Social Platform — Company Forms
# ============================================================================

from django import forms
from .models import CompanyProfile, Product, Service, CompanyReview


class CompanyProfileForm(forms.ModelForm):
    """Form for creating/editing a company profile."""

    class Meta:
        model = CompanyProfile
        fields = [
            "company_name", "description", "logo", "cover_image",
            "website", "email", "phone", "address", "city", "country",
            "specializations",
        ]
        widgets = {
            "company_name": forms.TextInput(attrs={"class": "form-input", "placeholder": "Company name"}),
            "description": forms.Textarea(attrs={"class": "form-textarea", "rows": 4, "placeholder": "About your company..."}),
            "website": forms.URLInput(attrs={"class": "form-input", "placeholder": "https://..."}),
            "email": forms.EmailInput(attrs={"class": "form-input", "placeholder": "contact@company.com"}),
            "phone": forms.TextInput(attrs={"class": "form-input", "placeholder": "+213..."}),
            "address": forms.Textarea(attrs={"class": "form-textarea", "rows": 2}),
            "city": forms.TextInput(attrs={"class": "form-input"}),
            "country": forms.TextInput(attrs={"class": "form-input"}),
            "specializations": forms.TextInput(attrs={"class": "form-input", "placeholder": "e.g. fertilizers, crop protection, seeds"}),
        }


class ProductForm(forms.ModelForm):
    """Form for adding/editing a product."""

    class Meta:
        model = Product
        fields = ["name", "description", "price", "currency", "image", "category"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-input", "placeholder": "Product name"}),
            "description": forms.Textarea(attrs={"class": "form-textarea", "rows": 3}),
            "price": forms.NumberInput(attrs={"class": "form-input", "step": "0.01"}),
            "currency": forms.TextInput(attrs={"class": "form-input", "value": "DA"}),
            "category": forms.Select(attrs={"class": "form-select"}),
        }


class ServiceForm(forms.ModelForm):
    """Form for adding/editing a service."""

    class Meta:
        model = Service
        fields = ["name", "description", "price_range", "service_area"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-input", "placeholder": "Service name"}),
            "description": forms.Textarea(attrs={"class": "form-textarea", "rows": 3}),
            "price_range": forms.TextInput(attrs={"class": "form-input", "placeholder": "e.g. 500-2000 DA"}),
            "service_area": forms.TextInput(attrs={"class": "form-input", "placeholder": "e.g. Algiers, Oran"}),
        }


class CompanyReviewForm(forms.ModelForm):
    """Form for submitting a company review."""

    class Meta:
        model = CompanyReview
        fields = ["rating", "review_text"]
        widgets = {
            "rating": forms.NumberInput(attrs={"class": "form-input", "min": 1, "max": 5}),
            "review_text": forms.Textarea(attrs={"class": "form-textarea", "rows": 3, "placeholder": "Share your experience..."}),
        }

    def clean_rating(self):
        rating = self.cleaned_data["rating"]
        if rating < 1 or rating > 5:
            raise forms.ValidationError("Rating must be between 1 and 5.")
        return rating
