# ============================================================================
# analysis/services/hybrid_ai_service.py
# Hybrid AI Orchestrator — Gemini + Local Fallback
# ============================================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HybridAIService:
    """
    Orchestrates between Gemini (online) and LocalAIAssistant (offline).

    Priority logic:
    1. Detect language from user message
    2. If scan_context is attached → include scan data in prompt
    3. Try Gemini first
    4. If Gemini fails → fall back to LocalAIAssistant
    """

    @staticmethod
    def _detect_language(message: str) -> str:
        """Detect if the message is primarily Arabic."""
        import re
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', message))
        latin_chars = len(re.findall(r'[a-zA-Z]', message))
        total = arabic_chars + latin_chars
        if total == 0:
            return "en"
        return "ar" if arabic_chars / total > 0.3 else "en"

    def chat(
        self,
        message: str,
        scan_context: Optional[Dict[str, Any]] = None,
        chat_context: Optional[List[Dict]] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a message through the hybrid AI pipeline.

        Args:
            message: User's text message
            scan_context: Optional dict with scan data (disease, ratio, stage, etc.)
            chat_context: Optional list of previous chat messages for Gemini context
            language: Optional language override ('ar' or 'en')

        Returns:
            dict with "text", "status", and optional "offline" flag
        """
        # Auto-detect language if not provided
        lang = language or self._detect_language(message)

        # ── Step 1: Try Gemini ───────────────────────────────────
        try:
            from .ai_service import get_gemini_service

            gemini = get_gemini_service()

            if gemini.is_available:
                # Build enriched prompt with scan context if available
                enriched_message = message
                if scan_context:
                    enriched_message = self._enrich_with_scan(message, scan_context, lang)

                result = gemini.chat(enriched_message, context=chat_context, language=lang)

                # Check if Gemini returned a valid response (not an error)
                if "text" in result and result.get("status") == "ok":
                    # If this was an offline fallback from Gemini, still return it
                    # but let the frontend know
                    if result.get("offline"):
                        logger.info("Gemini returned offline fallback — trying local AI instead")
                        # Fall through to local AI for better offline responses
                    else:
                        logger.info("Gemini response successful")
                        return result

        except Exception as e:
            logger.warning("Gemini service failed: %s — falling back to local AI", e)

        # ── Step 2: Fall back to Local AI ──────────────────────────
        try:
            from .local_ai_service import get_local_ai_service

            local_ai = get_local_ai_service()
            result = local_ai.chat(message, scan_context=scan_context, language=lang)
            logger.info("Local AI response generated (offline mode, lang=%s)", lang)
            return result

        except Exception as e:
            logger.error("Both Gemini and Local AI failed: %s", e)

            # Return error in appropriate language
            if lang == "ar":
                return {
                    "text": (
                        "🌿 **Nabtati**\n\n"
                        "عذراً، لا أستطيع معالجة طلبك الآن. "
                        "واجه كلا النظامين مشكلة.\n\n"
                        "**في هذه الأثناء:**\n"
                        "• استخدم صفحة التشخيص لفحص صور النباتات (تعمل بدون اتصال)\n"
                        "• راجع صفحة العلاج للتوصيات المعدة مسبقاً\n"
                        "• راجع فحوصاتك المحفوظة في ملفاتي\n\n"
                        "يرجى المحاولة مرة أخرى بعد لحظات."
                    ),
                    "status": "error",
                    "offline": True,
                }
            else:
                return {
                    "text": (
                        "🌿 **Nabtati**\n\n"
                        "I'm sorry, I'm unable to process your request right now. "
                        "Both the online AI and offline analysis systems encountered an issue.\n\n"
                        "**In the meantime:**\n"
                        "• Use the Diagnosis page to scan plant images (works offline)\n"
                        "• Check the Treatment page for pre-built recommendations\n"
                        "• Review your saved scans in My Files\n\n"
                        "Please try again in a few moments."
                    ),
                    "status": "error",
                    "offline": True,
                }

    def _enrich_with_scan(self, message: str, scan_context: Dict[str, Any],
                          language: str = "en") -> str:
        """Enrich the user's message with scan context for Gemini."""
        disease = scan_context.get("disease", "Unknown")
        ratio = scan_context.get("ratio", "N/A")
        stage = scan_context.get("stage", "N/A")
        confidence = scan_context.get("confidence", "N/A")
        recommendations = scan_context.get("recommendations", "")

        if language == "ar":
            enriched = (
                f"[سياق الفحص - المستخدم أرفق نتيجة فحص]\n"
                f"المرض المكتشف: {disease}\n"
                f"نسبة الإصابة: {ratio}%\n"
                f"مرحلة المرض: {stage}\n"
                f"ثقة الذكاء الاصطناعي: {confidence}%\n"
            )
            if recommendations:
                enriched += f"التوصيات السابقة: {recommendations}\n"
            enriched += (
                f"\nبناءً على بيانات الفحص هذه، أجب عن السؤال التالي باللغة العربية:\n"
                f"سؤال المستخدم: {message}\n\n"
                f"قدم نصيحة محددة وعملية مع مراعاة نتائج الفحص. "
                f"اذكر اسم المرض ومستوى الشدة والمرحلة في ردك. "
                f"أجب باللغة العربية فقط."
            )
        else:
            enriched = (
                f"[SCAN CONTEXT - The user has attached a scan result]\n"
                f"Disease Detected: {disease}\n"
                f"Infection Ratio: {ratio}%\n"
                f"Disease Stage: {stage}\n"
                f"AI Confidence: {confidence}%\n"
            )
            if recommendations:
                enriched += f"Previous Recommendations: {recommendations}\n"
            enriched += (
                f"\nBased on this scan data, answer the following question:\n"
                f"User Question: {message}\n\n"
                f"Provide specific, actionable advice considering the scan results above. "
                f"Reference the disease name, severity level, and stage in your response."
            )

        return enriched


# Singleton instance
_hybrid_instance = None


def get_hybrid_ai_service() -> HybridAIService:
    """Get or create the singleton HybridAIService instance."""
    global _hybrid_instance
    if _hybrid_instance is None:
        _hybrid_instance = HybridAIService()
    return _hybrid_instance
