# ============================================================================
# companies/views.py
# Agricultural Social Platform — Company Views & Messaging
# ============================================================================

import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Count
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import CompanyProfileForm, ProductForm, ServiceForm, CompanyReviewForm
from .models import (
    CompanyProfile, Product, Service, CompanyReview,
    CompanyConversation, CompanyMessage,
)
from .services import (
    get_or_create_company_conversation,
    send_company_message,
    mark_company_messages_read,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Company Directory
# ============================================================================

@login_required
def company_directory(request):
    """Browse and search all active companies."""
    companies = (
        CompanyProfile.objects
        .filter(is_active=True)
        .annotate(
            avg_rating=Avg("reviews__rating"),
            num_reviews=Count("reviews", distinct=True),
        )
        .order_by("-verified", "-avg_rating")
    )

    # Search
    q = request.GET.get("q", "").strip()
    if q:
        companies = companies.filter(
            Q(company_name__icontains=q) |
            Q(specializations__icontains=q) |
            Q(city__icontains=q)
        )

    paginator = Paginator(companies, 12)
    page = paginator.get_page(request.GET.get("page", 1))

    return render(request, "companies/directory.html", {
        "companies": page,
        "query": q,
    })


# ============================================================================
# Company Public Profile
# ============================================================================

@login_required
def company_profile(request, company_id):
    """Public company profile with products, services, and reviews."""
    company = get_object_or_404(
        CompanyProfile.objects.annotate(
            avg_rating=Avg("reviews__rating"),
        ),
        pk=company_id,
        is_active=True,
    )

    products = company.products.filter(is_active=True)
    services = company.services.filter(is_active=True)
    reviews = company.reviews.select_related("user").order_by("-created_at")[:20]

    # Check if user already reviewed
    user_reviewed = CompanyReview.objects.filter(
        company=company, user=request.user
    ).exists()

    review_form = CompanyReviewForm() if not user_reviewed else None

    return render(request, "companies/profile.html", {
        "company": company,
        "products": products,
        "services": services,
        "reviews": reviews,
        "user_reviewed": user_reviewed,
        "review_form": review_form,
    })


# ============================================================================
# Company Dashboard (Company Owner)
# ============================================================================

@login_required
def company_dashboard(request):
    """Company's own management dashboard."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        # If no profile exists, show creation form
        if request.method == "POST":
            form = CompanyProfileForm(request.POST, request.FILES)
            if form.is_valid():
                profile = form.save(commit=False)
                profile.user = request.user
                profile.save()
                # Set user role to company
                request.user.role = "company"
                request.user.save(update_fields=["role"])
                messages.success(request, "Company profile created!")
                return redirect("companies:dashboard")
        else:
            form = CompanyProfileForm()
        return render(request, "companies/dashboard.html", {
            "form": form,
            "creating": True,
        })

    # Edit existing profile
    if request.method == "POST":
        form = CompanyProfileForm(request.POST, request.FILES, instance=company)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated!")
            return redirect("companies:dashboard")
    else:
        form = CompanyProfileForm(instance=company)

    # Stats
    product_count = company.products.count()
    service_count = company.services.count()
    review_count = company.reviews.count()
    avg_rating = company.average_rating

    # Unread messages
    conversations = CompanyConversation.objects.filter(company=company)
    unread_total = sum(c.unread_count_for(request.user) for c in conversations)

    return render(request, "companies/dashboard.html", {
        "company": company,
        "form": form,
        "creating": False,
        "product_count": product_count,
        "service_count": service_count,
        "review_count": review_count,
        "avg_rating": avg_rating,
        "unread_total": unread_total,
    })


# ============================================================================
# Product Management
# ============================================================================

@login_required
def manage_products(request):
    """List and create products for the company."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        messages.error(request, "Create a company profile first.")
        return redirect("companies:dashboard")

    if request.method == "POST":
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            product = form.save(commit=False)
            product.company = company
            product.save()
            messages.success(request, f"Product '{product.name}' added!")
            return redirect("companies:manage_products")
    else:
        form = ProductForm()

    products = company.products.all()
    return render(request, "companies/manage_products.html", {
        "products": products,
        "form": form,
        "company": company,
    })


@login_required
@require_POST
def delete_product(request, product_id):
    """Delete a product (owner only)."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        return redirect("companies:dashboard")

    product = get_object_or_404(Product, pk=product_id, company=company)
    product.delete()
    messages.success(request, "Product deleted.")
    return redirect("companies:manage_products")


# ============================================================================
# Service Management
# ============================================================================

@login_required
def manage_services(request):
    """List and create services for the company."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        messages.error(request, "Create a company profile first.")
        return redirect("companies:dashboard")

    if request.method == "POST":
        form = ServiceForm(request.POST)
        if form.is_valid():
            service = form.save(commit=False)
            service.company = company
            service.save()
            messages.success(request, f"Service '{service.name}' added!")
            return redirect("companies:manage_services")
    else:
        form = ServiceForm()

    services = company.services.all()
    return render(request, "companies/manage_services.html", {
        "services": services,
        "form": form,
        "company": company,
    })


@login_required
@require_POST
def delete_service(request, service_id):
    """Delete a service (owner only)."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        return redirect("companies:dashboard")

    service = get_object_or_404(Service, pk=service_id, company=company)
    service.delete()
    messages.success(request, "Service deleted.")
    return redirect("companies:manage_services")


# ============================================================================
# Add Review
# ============================================================================

@login_required
@require_POST
def add_review(request, company_id):
    """Submit a review for a company."""
    company = get_object_or_404(CompanyProfile, pk=company_id)

    if CompanyReview.objects.filter(company=company, user=request.user).exists():
        messages.warning(request, "You've already reviewed this company.")
        return redirect("companies:profile", company_id=company.pk)

    form = CompanyReviewForm(request.POST)
    if form.is_valid():
        review = form.save(commit=False)
        review.company = company
        review.user = request.user
        review.save()
        messages.success(request, "Review submitted!")
    else:
        messages.error(request, "Invalid review data.")

    return redirect("companies:profile", company_id=company.pk)


# ============================================================================
# Company Messages Inbox (Company Side)
# ============================================================================

@login_required
def company_messages_inbox(request):
    """Company's message inbox — list all conversations with users."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        messages.error(request, "You need a company profile to access messages.")
        return redirect("companies:dashboard")

    conversations = (
        CompanyConversation.objects
        .filter(company=company)
        .select_related("user")
        .order_by("-updated_at")
    )

    conv_data = []
    for conv in conversations:
        last_msg = conv.last_message
        conv_data.append({
            "conversation": conv,
            "last_message": last_msg,
            "unread": conv.unread_count_for(request.user),
        })

    return render(request, "companies/company_messages.html", {
        "conversations": conv_data,
        "company": company,
    })


# ============================================================================
# Company Conversation View (Company Side)
# ============================================================================

@login_required
def company_conversation_view(request, conversation_id):
    """Full conversation thread between company and a user."""
    try:
        company = request.user.company_profile
    except CompanyProfile.DoesNotExist:
        return redirect("companies:dashboard")

    conversation = get_object_or_404(
        CompanyConversation, pk=conversation_id, company=company
    )

    # Mark messages as read
    mark_company_messages_read(request.user, conversation_id)

    msgs = conversation.messages.all().order_by("created_at")

    return render(request, "companies/company_conversation.html", {
        "conversation": conversation,
        "messages_list": msgs,
        "company": company,
        "other_user": conversation.user,
    })


# ============================================================================
# User's Company Messages (User Side)
# ============================================================================

@login_required
def user_company_messages(request):
    """User's list of company conversations."""
    conversations = (
        CompanyConversation.objects
        .filter(user=request.user)
        .select_related("company__user")
        .order_by("-updated_at")
    )

    conv_data = []
    for conv in conversations:
        last_msg = conv.last_message
        conv_data.append({
            "conversation": conv,
            "last_message": last_msg,
            "unread": conv.unread_count_for(request.user),
        })

    return render(request, "companies/user_company_messages.html", {
        "conversations": conv_data,
    })


# ============================================================================
# User's Conversation with a Company (User Side)
# ============================================================================

@login_required
def user_company_conversation(request, conversation_id):
    """User-side view of a single conversation with a company."""
    conversation = get_object_or_404(
        CompanyConversation, pk=conversation_id, user=request.user
    )

    # Mark messages as read
    mark_company_messages_read(request.user, conversation_id)

    msgs = conversation.messages.all().order_by("created_at")

    return render(request, "companies/user_company_conversation.html", {
        "conversation": conversation,
        "messages_list": msgs,
        "company": conversation.company,
    })


# ============================================================================
# Start Chat with Company
# ============================================================================

@login_required
def start_company_chat(request, company_id):
    """Initiate or resume conversation with a company."""
    company = get_object_or_404(CompanyProfile, pk=company_id, is_active=True)

    # Don't allow company to message itself
    if request.user == company.user:
        messages.warning(request, "You can't message your own company.")
        return redirect("companies:profile", company_id=company.pk)

    conv = get_or_create_company_conversation(request.user, company)
    return redirect("companies:user_conversation", conversation_id=conv.pk)


# ============================================================================
# API: Send Message
# ============================================================================

@login_required
@require_POST
def api_company_send_message(request):
    """AJAX: Send a message in a user↔company conversation."""
    # Always FormData (multipart) — read directly from POST/FILES
    conversation_id = request.POST.get("conversation_id", "").strip()
    text  = request.POST.get("text", "").strip()
    image = request.FILES.get("image")
    file  = request.FILES.get("file")

    if not conversation_id:
        return JsonResponse({"error": "Missing conversation_id."}, status=400)
    if not text and not image and not file:
        return JsonResponse({"error": "Message is empty."}, status=400)

    try:
        conv_id = int(conversation_id)
    except ValueError:
        return JsonResponse({"error": "Invalid conversation_id."}, status=400)

    success, result = send_company_message(
        sender=request.user,
        conversation_id=conv_id,
        text=text,
        image=image,
        file=file,
    )

    if not success:
        return JsonResponse({"error": result}, status=400)

    return JsonResponse({
        "status": "ok",
        "message": {
            "id":        result.pk,
            "text":      result.text or "",
            "sender":    result.sender.username,
            "is_me":     True,
            "image":     result.image.url if result.image else None,
            "file":      result.file.url  if result.file  else None,
            "file_name": result.file_name or "",
            "created_at": result.created_at.strftime("%H:%M"),
        },
    })



# ============================================================================
# API: Get Messages (Polling)
# ============================================================================

@login_required
def api_company_get_messages(request, conversation_id):
    """AJAX: Poll for messages in a conversation."""
    conversation = get_object_or_404(CompanyConversation, pk=conversation_id)

    # Verify user is part of this conversation
    if request.user != conversation.user and request.user != conversation.company.user:
        return JsonResponse({"error": "Not authorized."}, status=403)

    after_id = request.GET.get("after")
    msgs = conversation.messages.all().order_by("created_at")
    if after_id:
        msgs = msgs.filter(pk__gt=int(after_id))

    data = [
        {
            "id": m.pk,
            "text": m.text,
            "sender": m.sender.username,
            "is_me": m.sender == request.user,
            "image": m.image.url if m.image else None,
            "file": m.file.url if m.file else None,
            "file_name": m.file_name,
            "is_read": m.is_read,
            "created_at": m.created_at.strftime("%H:%M"),
        }
        for m in msgs
    ]

    return JsonResponse({"status": "ok", "messages": data})


# ============================================================================
# API: Mark Messages Read
# ============================================================================

@login_required
@require_POST
def api_company_mark_read(request, conversation_id):
    """AJAX: Mark all messages as read in a conversation."""
    mark_company_messages_read(request.user, conversation_id)
    return JsonResponse({"status": "ok"})


# ============================================================================
# API: Company Search
# ============================================================================

@login_required
def api_company_search(request):
    """AJAX: Search companies by name, specialization, or city."""
    q = request.GET.get("q", "").strip()
    if not q or len(q) < 2:
        return JsonResponse({"status": "ok", "companies": []})

    companies = (
        CompanyProfile.objects
        .filter(is_active=True)
        .filter(
            Q(company_name__icontains=q) |
            Q(specializations__icontains=q) |
            Q(city__icontains=q)
        )
        .order_by("-verified")[:10]
    )

    data = [
        {
            "id": c.pk,
            "name": c.company_name,
            "logo": c.logo.url if c.logo else None,
            "city": c.city,
            "verified": c.verified,
            "specializations": c.specializations,
        }
        for c in companies
    ]

    return JsonResponse({"status": "ok", "companies": data})

@login_required
@require_POST
def api_delete_message(request, message_id):
    """AJAX: Delete a message (sender only)."""
    msg = get_object_or_404(CompanyMessage, pk=message_id, sender=request.user)
    msg.delete()
    return JsonResponse({"status": "ok"})
