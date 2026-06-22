"""
Nabtati — Centralized Gemini AI Service
Provides chat, summarization, treatment planning, and comparison via Google Gemini.
"""

import json
import logging
import hashlib
from functools import lru_cache
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    GEMINI_MODE = "legacy"
except ImportError:
    try:
        from google import genai
        GEMINI_AVAILABLE = True
        GEMINI_MODE = "new"
    except ImportError:
        GEMINI_AVAILABLE = False
        GEMINI_MODE = None
        logger.warning("No Gemini SDK installed. AI features disabled.")


# ===== OFFLINE FALLBACKS =====
# Used when Gemini API is unavailable (quota exceeded, network error, etc.)
FALLBACK_RESPONSES = {
    "chat": (
        "🌿 **Nabtati (Offline Mode)**\n\n"
        "I'm currently running in offline mode due to API limits. "
        "Here are some general tips while I'm unavailable:\n\n"
        "• **For disease identification**: Upload a leaf image on the Diagnosis page — "
        "the ML model works offline and will identify the disease.\n"
        "• **For treatment info**: Visit the Treatment Plan page after a scan for "
        "pre-built fungicide recommendations.\n"
        "• **For monitoring**: Use the Compare feature in My Files to track disease progression.\n\n"
        "The AI chat will be back once API limits reset (usually within an hour). "
        "In the meantime, all image analysis and core diagnosis features continue to work!"
    ),
    "summary": (
        "This scan detected a plant disease with the parameters shown above. "
        "Review the confidence level and disease stage to assess urgency. "
        "For immediate treatment options, visit the Treatment Plan page. "
        "Save this scan to track disease progression over time."
    ),
    "treatment": (
        "**General Treatment Protocol (Offline)**\n\n"
        "**1. IMMEDIATE ACTIONS**\n"
        "• Isolate affected plants to prevent spread\n"
        "• Remove severely infected leaves and dispose properly\n"
        "• Ensure adequate air circulation between plants\n\n"
        "**2. SHORT-TERM (Days 1-3)**\n"
        "• Apply a broad-spectrum fungicide (e.g., Copper Oxychloride 2-3 kg/ha)\n"
        "• Reduce overhead irrigation — switch to drip if possible\n"
        "• Monitor neighboring plants for early symptoms\n\n"
        "**3. MEDIUM-TERM (Days 4-7)**\n"
        "• Apply a systemic fungicide (e.g., Azoxystrobin 0.5-1.0 L/ha)\n"
        "• Add foliar nutrition (potassium-rich fertilizer)\n"
        "• Re-assess severity and document progress\n\n"
        "**4. PREVENTION**\n"
        "• Practice crop rotation\n"
        "• Use resistant varieties when available\n"
        "• Maintain proper plant spacing\n\n"
        "⚠️ **Safety**: Always wear protective equipment when applying chemicals. "
        "Follow label instructions for dosage and withholding periods."
    ),
    "recommendations": (
        "**General Crop Health Recommendations (Offline)**\n\n"
        "**1. Regular Monitoring**\n"
        "• Inspect plants weekly for early disease signs\n"
        "• Use Nabtati scans to document and track changes\n\n"
        "**2. Preventive Care**\n"
        "• Maintain proper irrigation schedules\n"
        "• Ensure good air circulation and spacing\n"
        "• Apply preventive fungicides during humid seasons\n\n"
        "**3. Soil Health**\n"
        "• Test soil pH and nutrients annually\n"
        "• Add organic matter to improve drainage\n"
        "• Practice crop rotation every 2-3 seasons\n\n"
        "For personalized AI recommendations, try again when the API resets."
    ),
}

# Arabic fallback responses
FALLBACK_RESPONSES_AR = {
    "chat": (
        "🌿 **Nabtati (وضع عدم الاتصال)**\n\n"
        "أعمل حالياً في وضع عدم الاتصال بسبب حدود API. "
        "إليك بعض النصائح العامة في الوقت الحالي:\n\n"
        "• **لتحديد الأمراض**: ارفع صورة ورقة في صفحة التشخيص — "
        "نموذج التعلم الآلي يعمل بدون اتصال وسيحدد المرض.\n"
        "• **لمعلومات العلاج**: زر صفحة خطة العلاج بعد الفحص "
        "للحصول على توصيات المبيدات الفطرية.\n"
        "• **للمراقبة**: استخدم ميزة المقارنة في ملفاتي لتتبع تطور المرض.\n\n"
        "ستعود المحادثة الذكية عند إعادة تعيين حدود API (عادةً خلال ساعة). "
        "في هذه الأثناء، جميع ميزات تحليل الصور والتشخيص تعمل بشكل طبيعي!"
    ),
    "summary": (
        "اكتشف هذا الفحص مرضاً نباتياً بالمعايير الموضحة أعلاه. "
        "راجع مستوى الثقة ومرحلة المرض لتقييم مدى الاستعجال. "
        "لخيارات العلاج الفوري، زر صفحة خطة العلاج. "
        "احفظ هذا الفحص لتتبع تطور المرض بمرور الوقت."
    ),
    "treatment": (
        "**بروتوكول العلاج العام (وضع عدم الاتصال)**\n\n"
        "**1. إجراءات فورية**\n"
        "• اعزل النباتات المصابة لمنع الانتشار\n"
        "• أزل الأوراق المصابة بشدة وتخلص منها بشكل صحيح\n"
        "• تأكد من تهوية كافية بين النباتات\n\n"
        "**2. المدى القصير (الأيام 1-3)**\n"
        "• طبق مبيداً فطرياً واسع الطيف (مثل أوكسي كلوريد النحاس 2-3 كجم/هكتار)\n"
        "• قلل الري العلوي — انتقل للتنقيط إن أمكن\n"
        "• راقب النباتات المجاورة بحثاً عن أعراض مبكرة\n\n"
        "**3. المدى المتوسط (الأيام 4-7)**\n"
        "• طبق مبيداً فطرياً جهازياً (مثل أزوكسيستروبين 0.5-1.0 لتر/هكتار)\n"
        "• أضف تغذية ورقية (سماد غني بالبوتاسيوم)\n"
        "• أعد تقييم الشدة ووثق التقدم\n\n"
        "**4. الوقاية**\n"
        "• مارس تناوب المحاصيل\n"
        "• استخدم أصنافاً مقاومة عند توفرها\n"
        "• حافظ على التباعد المناسب بين النباتات\n\n"
        "⚠️ **السلامة**: ارتدِ دائماً معدات الحماية عند استخدام المواد الكيميائية. "
        "اتبع تعليمات الملصق للجرعات وفترات الانتظار."
    ),
    "recommendations": (
        "**توصيات صحة المحاصيل العامة (وضع عدم الاتصال)**\n\n"
        "**1. المراقبة المنتظمة**\n"
        "• افحص النباتات أسبوعياً بحثاً عن علامات مبكرة للأمراض\n"
        "• استخدم فحوصات Nabtati لتوثيق وتتبع التغييرات\n\n"
        "**2. الرعاية الوقائية**\n"
        "• حافظ على جداول ري مناسبة\n"
        "• تأكد من تهوية وتباعد جيدين\n"
        "• طبق مبيدات فطرية وقائية خلال المواسم الرطبة\n\n"
        "**3. صحة التربة**\n"
        "• افحص حموضة التربة والمغذيات سنوياً\n"
        "• أضف مواد عضوية لتحسين الصرف\n"
        "• مارس تناوب المحاصيل كل 2-3 مواسم\n\n"
        "للحصول على توصيات ذكية مخصصة، حاول مرة أخرى عند إعادة تعيين API."
    ),
}


class GeminiService:
    """Centralized service for all Gemini AI interactions."""

    _instance = None
    _model = None
    _cache = {}

    SYSTEM_PROMPT = (
        "You are Nabtati, an expert agronomist and plant pathologist assistant. "
        "You provide accurate, actionable advice about plant diseases, treatment strategies, "
        "irrigation plans, fungicide recommendations, and crop management. "
        "Keep responses concise, practical, and farmer-friendly. "
        "When discussing chemicals, always include safety warnings. "
        "Format responses with clear sections using line breaks."
    )

    SYSTEM_PROMPT_AR = (
        "أنت Nabtati، مساعد ذكي متخصص في أمراض النبات والزراعة. "
        "تقدم نصائح دقيقة وعملية حول أمراض النبات واستراتيجيات العلاج "
        "وخطط الري وتوصيات المبيدات الفطرية وإدارة المحاصيل. "
        "أجب دائماً باللغة العربية الفصحى الحديثة بأسلوب واضح وسهل الفهم. "
        "استخدم المصطلحات الزراعية العربية الدقيقة. "
        "عند ذكر المواد الكيميائية، اذكر دائماً تحذيرات السلامة. "
        "نسّق الردود بأقسام واضحة مع نقاط وعناوين. "
        "اذكر الأسماء العلمية بالإنجليزية بين قوسين عند الحاجة."
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return

        if not GEMINI_AVAILABLE:
            logger.error("Gemini SDK not available")
            return

        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            logger.error("GEMINI_API_KEY not set in settings")
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=self.SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=2048,
                ),
            )
            logger.info("Gemini AI service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._model = None

    @property
    def is_available(self):
        return self._model is not None

    def _cache_key(self, prefix, text):
        """Generate a cache key from the input text."""
        return f"{prefix}:{hashlib.md5(text.encode()).hexdigest()}"

    def _get_cached(self, key):
        """Get a cached response if available."""
        return self._cache.get(key)

    def _set_cached(self, key, value):
        """Store a response in the cache (max 200 entries)."""
        if len(self._cache) > 200:
            # Remove oldest entries
            oldest_keys = list(self._cache.keys())[:50]
            for k in oldest_keys:
                del self._cache[k]
        self._cache[key] = value

    def _get_fallback(self, cache_prefix, language="en"):
        """Return a fallback response for the given category."""
        if language == "ar":
            fallback_text = FALLBACK_RESPONSES_AR.get(
                cache_prefix, FALLBACK_RESPONSES_AR.get("chat", "")
            )
        else:
            fallback_text = FALLBACK_RESPONSES.get(
                cache_prefix, FALLBACK_RESPONSES.get("chat", "")
            )
        if fallback_text:
            return {"text": fallback_text, "status": "ok", "offline": True}
        return None

    def _generate(self, prompt, use_cache=True, cache_prefix="gen", language="en"):
        """Core generation method with error handling, caching, and fallbacks."""
        if not self.is_available:
            fallback = self._get_fallback(cache_prefix, language)
            if fallback:
                return fallback
            return {"error": "AI service is not available. Please check API configuration."}

        # Check cache
        if use_cache:
            cache_key = self._cache_key(cache_prefix, prompt)
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        try:
            response = self._model.generate_content(prompt)

            if response and response.text:
                result = {"text": response.text.strip(), "status": "ok"}
                if use_cache:
                    self._set_cached(cache_key, result)
                return result
            else:
                fallback = self._get_fallback(cache_prefix, language)
                if fallback:
                    return fallback
                return {"error": "AI returned an empty response. Please try again."}

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            error_msg = str(e)

            # For quota and connection errors, return useful fallback content
            if "quota" in error_msg.lower() or "resource" in error_msg.lower():
                fallback = self._get_fallback(cache_prefix, language)
                if fallback:
                    return fallback
                return {"error": "API quota exceeded. Please try again later."}
            elif "safety" in error_msg.lower():
                return {"error": "Content was filtered by safety settings. Please rephrase your question."}
            elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
                return {"error": "Invalid API key. Please check your configuration."}
            else:
                # For any other error, try fallback before showing error
                fallback = self._get_fallback(cache_prefix, language)
                if fallback:
                    return fallback
                return {"error": f"AI service error: {error_msg}"}

    def chat(self, message, context=None, language="en"):
        """
        Conversational AI for plant disease Q&A.

        Args:
            message: User's message
            context: Optional list of previous messages [{"role": "user"/"ai", "content": "..."}]
            language: Response language ('ar' for Arabic, 'en' for English)

        Returns:
            dict with "text" or "error"
        """
        conversation_context = ""
        if context:
            for msg in context[-6:]:  # Last 6 messages for context
                role = "User" if msg.get("role") == "user" else "Nabtati"
                conversation_context += f"{role}: {msg['content']}\n"

        if language == "ar":
            prompt = f"""محادثة أمراض النبات:

{conversation_context}المستخدم: {message}

{self.SYSTEM_PROMPT_AR}
قدم رداً مفيداً وموجزاً كـ Nabtati باللغة العربية الفصحى.
إذا كان السؤال خارج علم أمراض النبات، أعد توجيه المحادثة بلطف.
استخدم النقاط للتوصيات والعناوين للأقسام.
أجب باللغة العربية فقط."""
        else:
            prompt = f"""Plant disease conversation:

{conversation_context}User: {message}

Provide a helpful, concise response as Nabtati. If the question is outside plant pathology, 
politely redirect the conversation. Use bullet points for recommendations."""

        return self._generate(prompt, use_cache=False, cache_prefix="chat", language=language)

    def summarize_scan(self, scan_data):
        """
        Generate a human-readable summary of a scan result.

        Args:
            scan_data: dict with disease, confidence, ratio, stage, yield_loss

        Returns:
            dict with "text" or "error"
        """
        prompt = f"""Summarize this plant disease scan in 3-4 sentences for a farmer:

Disease: {scan_data.get('disease', 'Unknown')}
Confidence: {scan_data.get('confidence', 'N/A')}
Disease Severity: {scan_data.get('ratio', 'N/A')}%
Stage: {scan_data.get('stage', 'N/A')}
Yield Loss Estimate: {scan_data.get('yield_loss', 'N/A')}%

Include: severity assessment, urgency level, and one key recommendation.
Keep it short and actionable."""

        cache_text = json.dumps(scan_data, sort_keys=True)
        return self._generate(prompt, use_cache=True, cache_prefix="summary")

    def generate_treatment_plan(self, disease, stage, ratio):
        """
        Generate a detailed treatment plan.

        Args:
            disease: Disease name
            stage: Disease stage (Early, Moderate, Advanced)
            ratio: Infection ratio percentage

        Returns:
            dict with "text" or "error"
        """
        prompt = f"""Create a detailed 7-day treatment plan for:

Disease: {disease}
Stage: {stage}
Severity: {ratio}%

Structure as:
1. IMMEDIATE ACTIONS (today)
2. SHORT-TERM (days 1-3)
3. MEDIUM-TERM (days 4-7)
4. MONITORING PLAN
5. PREVENTION TIPS

Include specific fungicide recommendations with dosages.
Add safety warnings for chemical use.
Keep practical and farmer-friendly."""

        return self._generate(prompt, cache_prefix="treatment")

    def analyze_text(self, text):
        """
        General text analysis for crop notes or observations.

        Args:
            text: User's text to analyze

        Returns:
            dict with "text" or "error"
        """
        prompt = f"""Analyze the following crop observation and provide insights:

"{text}"

Provide:
1. Key observations identified
2. Possible issues or concerns
3. Recommended actions
4. Follow-up monitoring suggestions

Be specific and practical."""

        return self._generate(prompt, cache_prefix="analyze")

    def compare_and_advise(self, scan1_data, scan2_data):
        """
        Compare two scans and provide progression advice.

        Args:
            scan1_data: dict with disease, ratio, stage (older scan)
            scan2_data: dict with disease, ratio, stage (newer scan)

        Returns:
            dict with "text" or "error"
        """
        prompt = f"""Compare these two plant disease scans and analyze the progression:

PREVIOUS SCAN:
- Disease: {scan1_data.get('disease', 'Unknown')}
- Severity: {scan1_data.get('ratio', 'N/A')}%
- Stage: {scan1_data.get('stage', 'N/A')}

CURRENT SCAN:
- Disease: {scan2_data.get('disease', 'Unknown')}
- Severity: {scan2_data.get('ratio', 'N/A')}%
- Stage: {scan2_data.get('stage', 'N/A')}

Provide:
1. PROGRESSION ASSESSMENT (improving/worsening/stable)
2. RATE OF CHANGE analysis
3. CRITICAL WARNINGS if worsening
4. ADJUSTED RECOMMENDATIONS
5. PROGNOSIS if current trend continues

Be specific and urgent if the disease is worsening."""

        combined = json.dumps({**scan1_data, **scan2_data}, sort_keys=True)
        return self._generate(prompt, cache_prefix="compare")

    def get_recommendations(self, scan_history):
        """
        Smart recommendations based on scan history.

        Args:
            scan_history: list of recent scan dicts

        Returns:
            dict with "text" or "error"
        """
        history_text = ""
        for i, scan in enumerate(scan_history[:10], 1):
            history_text += (
                f"{i}. {scan.get('disease', 'Unknown')} — "
                f"Severity: {scan.get('ratio', 'N/A')}%, "
                f"Stage: {scan.get('stage', 'N/A')}, "
                f"Date: {scan.get('date', 'N/A')}\n"
            )

        prompt = f"""Based on this scan history, provide personalized recommendations:

SCAN HISTORY:
{history_text}

Provide:
1. PATTERNS DETECTED (recurring diseases, trends)
2. TOP 3 PRIORITY ACTIONS
3. SEASONAL RECOMMENDATIONS
4. PREVENTIVE MEASURES for most common issues
5. LONG-TERM CROP HEALTH STRATEGY

Be specific to the detected patterns."""

        return self._generate(prompt, cache_prefix="reco")


# Singleton instance
def get_gemini_service():
    """Get or create the singleton GeminiService instance."""
    return GeminiService()
