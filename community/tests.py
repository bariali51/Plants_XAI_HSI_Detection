# ============================================================================
# community/tests.py
# Agricultural Social Platform — Community Tests
# ============================================================================

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth import get_user_model
from community.models import Post, PromotedPost

User = get_user_model()

class CommunityAdTests(TestCase):
    def setUp(self):
        # Create test users
        self.user = User.objects.create_user(
            username="farmer1",
            password="password123",
            email="farmer1@example.com",
            role="user"
        )
        self.company_user = User.objects.create_user(
            username="cropcorp",
            password="password123",
            email="cropcorp@example.com",
            role="company"
        )
        # Create regular post
        self.post = Post.objects.create(
            author=self.user,
            title="Regular Post Title",
            body="This is a regular post body content.",
            post_type="question"
        )
        # Create a post to be promoted
        self.promo_post = Post.objects.create(
            author=self.company_user,
            title="Promoted Post Title",
            body="This is a promoted post body content.",
            post_type="solution"
        )
        # Create promotion
        self.promotion = PromotedPost.objects.create(
            post=self.promo_post,
            promoted_by=self.company_user,
            duration_days=7,
            amount=50.00,
            payment_method="ccp",
            account_number="12345678",
            status="pending"
        )

    def test_default_is_ad_is_false(self):
        """Verify is_ad defaults to False for a new post."""
        self.assertFalse(self.post.is_ad)
        self.assertFalse(self.promo_post.is_ad)

    def test_activation_sets_is_ad_to_true(self):
        """Verify is_ad is set to True upon promotion activation."""
        self.promotion.activate()
        # Refresh from DB
        self.promo_post.refresh_from_db()
        self.assertTrue(self.promo_post.is_ad)

    def test_expire_if_due_resets_is_ad(self):
        """Verify is_ad is reset to False when expire_if_due expires the promotion."""
        self.promotion.activate()
        self.promo_post.refresh_from_db()
        self.assertTrue(self.promo_post.is_ad)

        # Set end_date to the past to simulate expiration
        self.promotion.end_date = timezone.now() - timedelta(minutes=5)
        self.promotion.save()

        expired = self.promotion.expire_if_due()
        self.assertTrue(expired)
        
        self.promotion.refresh_from_db()
        self.assertEqual(self.promotion.status, "expired")
        
        self.promo_post.refresh_from_db()
        self.assertFalse(self.promo_post.is_ad)

    def test_feed_view_auto_expires_promotions(self):
        """Verify feed view bulk expire updates both status and is_ad field."""
        self.promotion.activate()
        self.promo_post.refresh_from_db()
        self.assertTrue(self.promo_post.is_ad)

        # Force the promotion to expire by moving the end_date in the database
        self.promotion.end_date = timezone.now() - timedelta(minutes=5)
        self.promotion.save()

        # Log in and request the feed view
        self.client.login(username="farmer1", password="password123")
        response = self.client.get(reverse("community:feed"))
        self.assertEqual(response.status_code, 200)

        # Check that post's is_ad has been set to False
        self.promo_post.refresh_from_db()
        self.assertFalse(self.promo_post.is_ad)
        
        self.promotion.refresh_from_db()
        self.assertEqual(self.promotion.status, "expired")

    def test_ad_badge_renders_in_feed_and_detail(self):
        """Verify Ad badge is present in the feed and details page if post is_ad is True."""
        self.promotion.activate()
        self.promo_post.refresh_from_db()
        self.assertTrue(self.promo_post.is_ad)

        self.client.login(username="farmer1", password="password123")

        # 1. Check Feed
        response = self.client.get(reverse("community:feed"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'is-ad-card')
        self.assertContains(response, 'premium-ad-badge-title')
        self.assertContains(response, 'Ad')

        # 2. Check Detail Page
        response = self.client.get(reverse("community:post_detail", args=[self.promo_post.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'is-ad-card')
        self.assertContains(response, 'premium-ad-badge-title')
        self.assertContains(response, 'Ad')

        # 3. Check Search Results page
        response = self.client.get(reverse("community:search") + "?q=Promoted")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'is-ad-card')
        self.assertContains(response, 'premium-ad-badge-title')
        self.assertContains(response, 'Ad')

    def test_recommend_experts_works_on_sqlite(self):
        """Verify recommend_experts_for_post works without throwing NotSupportedError on SQLite."""
        from experts.models import ExpertProfile
        from community.models import Tag
        from community.recommendations import recommend_experts_for_post

        # Create expert user and profile
        expert_user = User.objects.create_user(
            username="expert1",
            password="password123",
            email="expert1@example.com",
            role="expert"
        )
        ExpertProfile.objects.create(
            user=expert_user,
            specialization="Plant Pathology",
            skills=["Tomato", "Fungus"],
            is_active=True
        )

        # Create tag
        tag = Tag.objects.create(name="Tomato", category="crop")
        self.promo_post.tags.add(tag)

        # Call the recommendation function
        recommendations = recommend_experts_for_post(self.promo_post)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["id"], expert_user.pk)
        self.assertEqual(recommendations[0]["name"], "expert1")

