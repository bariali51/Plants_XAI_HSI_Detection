# ============================================================================
# analysis/services/local_ai_service.py
# Rule-Based Local AI Assistant — Fully Offline NLP System
# Supports English + Arabic (العربية)
# ============================================================================

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Knowledge Base at module level
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KNOWLEDGE_PATH = os.path.join(_BASE_DIR, "model_files", "plant_knowledge.json")
_TREATMENTS_PATH = os.path.join(_BASE_DIR, "model_files", "treatments.json")
_CLASS_INDICES_PATH = os.path.join(_BASE_DIR, "model_files", "class_indices.json")

with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as _f:
    _KNOWLEDGE = json.load(_f)

with open(_TREATMENTS_PATH, "r", encoding="utf-8") as _f:
    _TREATMENTS_DATA = json.load(_f)

with open(_CLASS_INDICES_PATH, "r", encoding="utf-8") as _f:
    _CLASS_INDICES = json.load(_f)

# Pre-process data for fast lookups
_SYNONYMS: Dict[str, str] = _KNOWLEDGE["keyword_synonyms"]
_PLANT_ALIASES: Dict[str, List[str]] = _KNOWLEDGE["plant_aliases"]
_NUTRIENT_DATA: Dict[str, dict] = _KNOWLEDGE["nutrient_deficiencies"]
_DISEASE_KEYWORDS: Dict[str, List[str]] = _KNOWLEDGE["disease_keywords"]
_GENERAL_ADVICE: Dict[str, List[str]] = _KNOWLEDGE["general_advice"]
_GENERAL_ADVICE_AR: Dict[str, List[str]] = _KNOWLEDGE.get("general_advice_ar", {})
_DISEASE_NAMES_AR: Dict[str, str] = _KNOWLEDGE.get("disease_names_ar", {})
_STOPWORDS: set = set(_KNOWLEDGE["stopwords"])
_TREATMENT_RECS: Dict[str, List[str]] = _TREATMENTS_DATA["TREATMENT_RECOMMENDATIONS"]

# Build reverse plant alias index: alias -> canonical name
_ALIAS_INDEX: Dict[str, str] = {}
for canonical, aliases in _PLANT_ALIASES.items():
    for alias in aliases:
        _ALIAS_INDEX[alias.lower()] = canonical

# Build disease label index from class_indices.json
_DISEASE_LABELS: List[str] = list(_CLASS_INDICES.keys())
_DISEASE_LABELS_LOWER: Dict[str, str] = {
    label.lower().replace(" ", "_"): label for label in _DISEASE_LABELS
}

# Arabic plant name lookup (Arabic -> English canonical)
_PLANT_NAMES_AR: Dict[str, str] = {
    "طماطم": "tomato", "بندورة": "tomato",
    "بطاطا": "potato", "بطاطس": "potato",
    "أرز": "rice", "قمح": "wheat", "ذرة": "corn",
    "صويا": "soybean", "تفاح": "apple", "عنب": "grape",
    "كرز": "cherry", "خوخ": "peach", "فراولة": "strawberry",
    "فلفل": "pepper", "قهوة": "coffee", "خيار": "cucumber",
    "ليمون": "lemon", "مانجو": "mango", "زيتون": "olive",
}

# Arabic plant name reverse lookup (English -> Arabic)
_PLANT_NAMES_EN_TO_AR: Dict[str, str] = {
    "tomato": "الطماطم", "potato": "البطاطا", "rice": "الأرز",
    "wheat": "القمح", "corn": "الذرة", "soybean": "الصويا",
    "apple": "التفاح", "grape": "العنب", "cherry": "الكرز",
    "peach": "الخوخ", "strawberry": "الفراولة", "pepper": "الفلفل",
    "coffee": "القهوة", "cucumber": "الخيار", "lemon": "الليمون",
    "mango": "المانجو", "olive": "الزيتون",
}

logger.info(
    "LocalAI knowledge loaded: %d synonyms, %d plants, %d nutrients, %d diseases",
    len(_SYNONYMS), len(_PLANT_ALIASES), len(_NUTRIENT_DATA), len(_DISEASE_LABELS),
)


# ============================================================================
# Local AI Assistant
# ============================================================================

class LocalAIAssistant:
    """
    Rule-based NLP assistant that works fully offline.
    Supports English + Arabic (العربية).

    Pipeline:
    1. Detect language (Arabic / English)
    2. Tokenize + clean user message
    3. Normalize keywords (synonym expansion, typo correction)
    4. Match against diseases, nutrients, plants
    5. Build structured response in detected language
    """

    # ── Language Detection ───────────────────────────────────────────

    @staticmethod
    def _detect_language(message: str) -> str:
        """
        Detect if the message is primarily Arabic or English.
        Returns 'ar' or 'en'.
        """
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', message))
        latin_chars = len(re.findall(r'[a-zA-Z]', message))
        total = arabic_chars + latin_chars
        if total == 0:
            return "en"
        return "ar" if arabic_chars / total > 0.3 else "en"

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        """
        Normalize Arabic text:
        - Remove tashkeel (diacritics)
        - Normalize hamzas (أإآ -> ا)
        - Normalize ta marbuta (ة -> ه) for matching
        - Normalize alef maqsura (ى -> ي)
        """
        # Remove tashkeel
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        # Normalize hamzas
        text = re.sub(r'[أإآ]', 'ا', text)
        # Don't normalize ta marbuta and alef maqsura for general text,
        # but do it for keyword matching
        return text

    def chat(self, message: str, scan_context: Optional[Dict] = None,
             language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message and return a structured response.

        Args:
            message: User's text message
            scan_context: Optional dict with scan data (disease, ratio, stage, etc.)
            language: Optional language override ('ar' or 'en')

        Returns:
            dict with "text", "status", "offline" keys
        """
        # Detect language
        lang = language or self._detect_language(message)

        # If scan context is provided, prioritize it
        if scan_context:
            return self._respond_with_scan_context(message, scan_context, lang)

        # Extract and normalize keywords
        raw_keywords = self._extract_keywords(message)
        normalized = self._normalize_keywords(raw_keywords)

        # Match against knowledge base
        matched_plants = self._match_plants(normalized)
        matched_diseases = self._match_diseases(normalized, matched_plants)
        matched_nutrients = self._match_nutrients(normalized)

        # Detect intent
        intent = self._detect_intent(normalized, message.lower())

        # Build response in appropriate language
        if lang == "ar":
            response = self._build_arabic_response(
                original_message=message,
                keywords=normalized,
                plants=matched_plants,
                diseases=matched_diseases,
                nutrients=matched_nutrients,
                intent=intent,
            )
        else:
            response = self._build_response(
                original_message=message,
                keywords=normalized,
                plants=matched_plants,
                diseases=matched_diseases,
                nutrients=matched_nutrients,
                intent=intent,
            )

        return {"text": response, "status": "ok", "offline": True}

    # ── Keyword Extraction ───────────────────────────────────────────

    def _extract_keywords(self, message: str) -> List[str]:
        """Tokenize and extract meaningful keywords from the message."""
        # Lowercase, remove punctuation except underscores
        text = message.lower()
        # Normalize Arabic text
        text = self._normalize_arabic(text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = text.split()

        # Remove stopwords
        keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

        return keywords

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """Normalize keywords using synonym mappings."""
        normalized = []
        for kw in keywords:
            # Direct synonym match
            if kw in _SYNONYMS:
                normalized.append(_SYNONYMS[kw])
            else:
                # Fuzzy partial match for synonyms
                matched = False
                for syn_key, syn_val in _SYNONYMS.items():
                    if len(syn_key) >= 4 and (syn_key in kw or kw in syn_key):
                        normalized.append(syn_val)
                        matched = True
                        break
                if not matched:
                    normalized.append(kw)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for n in normalized:
            if n not in seen:
                seen.add(n)
                deduped.append(n)

        return deduped

    # ── Matching ─────────────────────────────────────────────────────

    def _match_plants(self, keywords: List[str]) -> List[str]:
        """Match keywords against known plant names."""
        matched = []
        for kw in keywords:
            if kw in _ALIAS_INDEX:
                plant = _ALIAS_INDEX[kw]
                if plant not in matched:
                    matched.append(plant)
        return matched

    def _match_diseases(
        self, keywords: List[str], plants: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Match keywords against disease patterns.

        Returns list of (disease_label, disease_key, confidence_score) tuples.
        """
        matches = []
        keyword_set = set(keywords)

        for disease_key, disease_keywords in _DISEASE_KEYWORDS.items():
            overlap = keyword_set & set(disease_keywords)
            if overlap:
                score = len(overlap) / len(disease_keywords)

                # Find matching class labels
                for label_key, label_name in _DISEASE_LABELS_LOWER.items():
                    # Check if disease key is in label
                    dk_clean = disease_key.replace("_", "")
                    lk_clean = label_key.replace("_", "")
                    if dk_clean in lk_clean or disease_key in label_key:
                        # If plants specified, filter by plant
                        if plants:
                            plant_match = any(p in label_key for p in plants)
                            if plant_match:
                                matches.append((label_name, disease_key, score + 0.2))
                        else:
                            matches.append((label_name, disease_key, score))

        # If no class-label match found but disease keywords match, return generic
        if not matches:
            for disease_key, disease_keywords in _DISEASE_KEYWORDS.items():
                overlap = keyword_set & set(disease_keywords)
                if overlap:
                    score = len(overlap) / len(disease_keywords)
                    display_name = disease_key.replace("_", " ").title()
                    matches.append((display_name, disease_key, score))

        # Sort by confidence and deduplicate
        matches.sort(key=lambda x: x[2], reverse=True)
        seen = set()
        unique = []
        for m in matches:
            if m[0] not in seen:
                seen.add(m[0])
                unique.append(m)
        return unique[:5]  # Top 5 matches

    def _match_nutrients(self, keywords: List[str]) -> List[Tuple[str, dict]]:
        """Match keywords against nutrient deficiency patterns."""
        matches = []
        keyword_set = set(keywords)

        # Check for deficiency-related keywords
        has_deficiency_intent = bool(
            keyword_set & {"deficiency", "lacking", "missing", "deficit", "shortage", "need"}
        )

        for nutrient_name, nutrient_data in _NUTRIENT_DATA.items():
            nutrient_keywords = set(nutrient_data["keywords"])
            overlap = keyword_set & nutrient_keywords

            if overlap:
                matches.append((nutrient_name, nutrient_data))
            elif has_deficiency_intent and nutrient_name in keyword_set:
                matches.append((nutrient_name, nutrient_data))

        return matches

    def _detect_intent(self, keywords: List[str], message: str) -> str:
        """Detect the user's intent category."""
        kw_set = set(keywords)

        if kw_set & {"treatment", "treat", "cure", "fix", "remedy", "spray", "fungicide"}:
            return "treatment"
        elif kw_set & {"water", "irrigation", "watering", "moisture", "drip"}:
            return "watering"
        elif kw_set & {"prevent", "prevention", "protect", "avoid", "stop"}:
            return "prevention"
        elif kw_set & {"identify", "diagnose", "diagnosis", "detect", "scan", "check"}:
            return "diagnosis"
        elif kw_set & {"soil", "fertilizer", "compost", "nutrient", "ph"}:
            return "soil_health"
        elif kw_set & {"deficiency", "lacking", "missing", "deficit"}:
            return "nutrient"
        elif "?" in message or any(w in message for w in ["what", "how", "why", "when"]):
            return "question"
        else:
            return "general"

    # ── English Response Building ────────────────────────────────────

    def _build_response(
        self,
        original_message: str,
        keywords: List[str],
        plants: List[str],
        diseases: List[Tuple[str, str, float]],
        nutrients: List[Tuple[str, dict]],
        intent: str,
    ) -> str:
        """Build a structured natural-language response in English."""
        sections = []

        sections.append(
            "🌿 **Nabtati (Offline Mode)**\n"
            "_I'm running locally with rule-based analysis. "
            "For more detailed AI responses, the online mode will resume when available._\n"
        )

        # ── Nutrient deficiency section
        if nutrients:
            sections.append("---\n**🧪 Nutrient Analysis**\n")
            for nutrient_name, data in nutrients:
                sections.append(f"**{nutrient_name.title()} Deficiency Detected**\n")
                sections.append("**Symptoms to look for:**")
                for s in data["symptoms"]:
                    sections.append(f"• {s}")
                sections.append("\n**Recommended actions:**")
                for a in data["advice"]:
                    sections.append(f"• {a}")
                sections.append("")

        # ── Disease match section
        if diseases:
            sections.append("---\n**🔬 Possible Disease Matches**\n")
            for label, key, score in diseases[:3]:
                confidence_label = "High" if score > 0.5 else "Moderate" if score > 0.3 else "Low"
                sections.append(
                    f"**{label}** (keyword match: {confidence_label})"
                )

                # Get treatments for this disease
                treatment_key = label.lower().strip().replace(" ", "_")
                treatments = _TREATMENT_RECS.get(treatment_key)
                if not treatments:
                    # Try partial match
                    for tk, tv in _TREATMENT_RECS.items():
                        if key in tk or tk in treatment_key:
                            treatments = tv
                            break

                if treatments:
                    sections.append("**Treatment recommendations:**")
                    for t in treatments:
                        sections.append(f"• {t}")
                sections.append("")

        # ── Plant-specific tips
        if plants and not diseases and not nutrients:
            sections.append("---\n**🌱 Plant Information**\n")
            for plant in plants:
                sections.append(f"**{plant.title()}**")
                # Find related diseases
                related = [
                    label for lk, label in _DISEASE_LABELS_LOWER.items()
                    if plant in lk
                ]
                if related:
                    sections.append("Known diseases for this plant:")
                    for r in related[:6]:
                        sections.append(f"• {r}")
                sections.append("")

        # ── Intent-based general advice
        if not diseases and not nutrients:
            if intent in _GENERAL_ADVICE:
                sections.append(f"---\n**💡 {intent.replace('_', ' ').title()} Tips**\n")
                for tip in _GENERAL_ADVICE[intent]:
                    sections.append(f"• {tip}")
                sections.append("")
            elif intent == "question" or intent == "general":
                # Provide general monitoring tips if nothing specific matched
                if not plants:
                    sections.append("---\n**💡 General Recommendations**\n")
                    for tip in _GENERAL_ADVICE.get("monitoring", []):
                        sections.append(f"• {tip}")
                    sections.append(
                        "\n_💡 Tip: For better results, try including the plant name "
                        "and specific symptoms in your question. For example: "
                        '"tomato yellow leaves" or "potato brown spots"._'
                    )

        # ── Footer
        sections.append(
            "\n---\n_📡 This response was generated offline. "
            "For AI-powered detailed analysis, the Gemini service will "
            "reconnect automatically._"
        )

        return "\n".join(sections)

    # ── Arabic Response Building ─────────────────────────────────────

    def _build_arabic_response(
        self,
        original_message: str,
        keywords: List[str],
        plants: List[str],
        diseases: List[Tuple[str, str, float]],
        nutrients: List[Tuple[str, dict]],
        intent: str,
    ) -> str:
        """Build a structured natural-language response in Arabic (MSA)."""
        sections = []

        sections.append(
            "🌿 **Nabtati (وضع عدم الاتصال)**\n"
            "_أعمل حالياً في الوضع المحلي بالتحليل القائم على القواعد. "
            "للحصول على ردود أكثر تفصيلاً، سيعود الوضع المتصل عند توفره._\n"
        )

        # ── Nutrient deficiency section (Arabic)
        if nutrients:
            sections.append("---\n**🧪 تحليل العناصر الغذائية**\n")
            for nutrient_name, data in nutrients:
                ar_name = {
                    "nitrogen": "النيتروجين", "phosphorus": "الفوسفور",
                    "potassium": "البوتاسيوم", "calcium": "الكالسيوم",
                    "magnesium": "المغنيسيوم", "iron": "الحديد",
                    "sulfur": "الكبريت",
                }.get(nutrient_name, nutrient_name.title())

                sections.append(f"**تم اكتشاف نقص {ar_name}**\n")

                # Use Arabic symptoms if available
                symptoms = data.get("symptoms_ar", data["symptoms"])
                sections.append("**الأعراض التي يجب البحث عنها:**")
                for s in symptoms:
                    sections.append(f"• {s}")

                # Use Arabic advice if available
                advice = data.get("advice_ar", data["advice"])
                sections.append("\n**الإجراءات الموصى بها:**")
                for a in advice:
                    sections.append(f"• {a}")
                sections.append("")

        # ── Disease match section (Arabic)
        if diseases:
            sections.append("---\n**🔬 الأمراض المحتملة**\n")
            for label, key, score in diseases[:3]:
                confidence_label = "عالية" if score > 0.5 else "متوسطة" if score > 0.3 else "منخفضة"

                # Get Arabic disease name
                ar_disease_name = _DISEASE_NAMES_AR.get(key, label)
                sections.append(
                    f"**{ar_disease_name}** (درجة التطابق: {confidence_label})"
                )

                # Get treatments for this disease
                treatment_key = label.lower().strip().replace(" ", "_")
                treatments = _TREATMENT_RECS.get(treatment_key)
                if not treatments:
                    for tk, tv in _TREATMENT_RECS.items():
                        if key in tk or tk in treatment_key:
                            treatments = tv
                            break

                if treatments:
                    sections.append("**توصيات العلاج:**")
                    for t in treatments:
                        sections.append(f"• {t}")
                sections.append("")

        # ── Plant-specific tips (Arabic)
        if plants and not diseases and not nutrients:
            sections.append("---\n**🌱 معلومات النبات**\n")
            for plant in plants:
                ar_plant_name = _PLANT_NAMES_EN_TO_AR.get(plant, plant.title())
                sections.append(f"**{ar_plant_name}**")
                related = [
                    label for lk, label in _DISEASE_LABELS_LOWER.items()
                    if plant in lk
                ]
                if related:
                    sections.append("الأمراض المعروفة لهذا النبات:")
                    for r in related[:6]:
                        # Try to get Arabic name
                        r_key = r.lower().replace(" ", "_")
                        ar_name = _DISEASE_NAMES_AR.get(r_key, r)
                        sections.append(f"• {ar_name}")
                sections.append("")

        # ── Intent-based general advice (Arabic)
        if not diseases and not nutrients:
            intent_titles_ar = {
                "watering": "💧 نصائح الري",
                "prevention": "🛡️ نصائح الوقاية",
                "soil_health": "🌍 صحة التربة",
                "monitoring": "👁️ المراقبة",
                "treatment": "💊 العلاج",
                "diagnosis": "🔍 التشخيص",
            }

            if intent in _GENERAL_ADVICE_AR:
                title = intent_titles_ar.get(intent, f"💡 {intent}")
                sections.append(f"---\n**{title}**\n")
                for tip in _GENERAL_ADVICE_AR[intent]:
                    sections.append(f"• {tip}")
                sections.append("")
            elif intent in ("question", "general"):
                if not plants:
                    sections.append("---\n**💡 توصيات عامة**\n")
                    for tip in _GENERAL_ADVICE_AR.get("monitoring", _GENERAL_ADVICE.get("monitoring", [])):
                        sections.append(f"• {tip}")
                    sections.append(
                        "\n_💡 نصيحة: للحصول على نتائج أفضل، حاول ذكر اسم النبات "
                        "والأعراض المحددة في سؤالك. مثال: "
                        '"طماطم أوراق صفراء" أو "بطاطا بقع بنية"._'
                    )

        # ── Footer (Arabic)
        sections.append(
            "\n---\n_📡 تم إنشاء هذه الاستجابة في وضع عدم الاتصال. "
            "سيعاود الاتصال بخدمة Gemini AI تلقائياً._"
        )

        return "\n".join(sections)

    # ── Scan Context Response ────────────────────────────────────────

    def _respond_with_scan_context(
        self, message: str, scan_context: Dict[str, Any],
        lang: str = "en"
    ) -> Dict[str, Any]:
        """Generate a response based on attached scan data."""
        disease = str(scan_context.get("disease", "Unknown"))
        ratio = scan_context.get("ratio", 0)
        stage = str(scan_context.get("stage", "Unknown"))
        confidence = scan_context.get("confidence", 0)

        # Determine severity
        try:
            ratio_val = float(str(ratio).replace("%", ""))
        except (ValueError, TypeError):
            ratio_val = 0

        if lang == "ar":
            return self._build_arabic_scan_response(
                message, disease, ratio_val, stage, confidence, scan_context
            )
        else:
            return self._build_english_scan_response(
                message, disease, ratio_val, stage, confidence, scan_context
            )

    def _build_english_scan_response(
        self, message: str, disease: str, ratio_val: float,
        stage: str, confidence, scan_context: Dict
    ) -> Dict[str, Any]:
        """Build English scan context response."""
        if ratio_val >= 60:
            severity = "critical"
            urgency = "⚠️ **URGENT**: Immediate action required!"
        elif ratio_val >= 40:
            severity = "advanced"
            urgency = "⚡ **Action needed within 24-48 hours.**"
        elif ratio_val >= 20:
            severity = "moderate"
            urgency = "📋 Begin treatment within the next few days."
        else:
            severity = "early"
            urgency = "👀 Monitor closely and apply preventive measures."

        sections = [
            "🌿 **Nabtati (Offline Mode) — Scan Analysis**\n",
            f"📊 **Scan Context:**",
            f"• Disease: **{disease}**",
            f"• Severity: **{ratio_val:.1f}%** ({stage})",
            f"• Confidence: **{confidence}%**",
            f"\n{urgency}\n",
        ]

        # Get treatments
        treatment_key = disease.lower().strip().replace(" ", "_")
        treatments = _TREATMENT_RECS.get(treatment_key)
        if not treatments:
            for tk, tv in _TREATMENT_RECS.items():
                if treatment_key in tk or tk in treatment_key:
                    treatments = tv
                    break

        if treatments:
            sections.append("---\n**💊 Treatment Plan:**\n")
            for i, t in enumerate(treatments, 1):
                sections.append(f"{i}. {t}")

        # Severity-specific advice
        sections.append(f"\n---\n**📋 Severity-Based Action ({severity.title()}):**\n")
        if severity == "critical":
            sections.extend([
                "• Isolate affected plants immediately",
                "• Remove and destroy severely infected tissue",
                "• Apply systemic fungicide within 24 hours",
                "• Consult agricultural specialist urgently",
                "• Monitor surrounding plants daily",
            ])
        elif severity == "advanced":
            sections.extend([
                "• Begin targeted fungicide treatment",
                "• Improve air circulation around plants",
                "• Adjust irrigation to reduce leaf wetness",
                "• Schedule follow-up scan in 3 days",
            ])
        elif severity == "moderate":
            sections.extend([
                "• Apply preventive fungicide",
                "• Monitor plant daily for progression",
                "• Ensure proper nutrient supply",
                "• Re-scan in 5-7 days to track changes",
            ])
        else:
            sections.extend([
                "• Continue regular monitoring",
                "• Apply preventive bio-fungicide",
                "• Maintain optimal growing conditions",
                "• Document with follow-up scans",
            ])

        # Answer user's specific question context
        user_question = message.lower().strip()
        if user_question and user_question not in ("", "what should i do now?", "help"):
            keywords = self._extract_keywords(message)
            normalized = self._normalize_keywords(keywords)
            nutrients = self._match_nutrients(normalized)

            if nutrients:
                sections.append("\n---\n**🧪 Additional — Nutrient Analysis:**\n")
                for nutrient_name, data in nutrients:
                    sections.append(f"**{nutrient_name.title()}:**")
                    for a in data["advice"][:3]:
                        sections.append(f"• {a}")

        sections.append(
            "\n---\n_📡 This response was generated offline using your scan data._"
        )

        return {"text": "\n".join(sections), "status": "ok", "offline": True}

    def _build_arabic_scan_response(
        self, message: str, disease: str, ratio_val: float,
        stage: str, confidence, scan_context: Dict
    ) -> Dict[str, Any]:
        """Build Arabic scan context response."""
        # Arabic severity labels
        if ratio_val >= 60:
            severity_ar = "حرج"
            urgency = "⚠️ **عاجل**: يلزم اتخاذ إجراء فوري!"
        elif ratio_val >= 40:
            severity_ar = "متقدم"
            urgency = "⚡ **يلزم اتخاذ إجراء خلال 24-48 ساعة.**"
        elif ratio_val >= 20:
            severity_ar = "متوسط"
            urgency = "📋 ابدأ العلاج خلال الأيام القليلة القادمة."
        else:
            severity_ar = "مبكر"
            urgency = "👀 راقب عن كثب وطبق إجراءات وقائية."

        # Arabic stage translation
        stage_ar = {"Early": "مبكر", "Moderate": "متوسط", "Advanced": "متقدم",
                    "Severe": "شديد", "Critical": "حرج"}.get(stage, stage)

        # Try to get Arabic disease name
        disease_key = disease.lower().strip().replace(" ", "_")
        ar_disease = _DISEASE_NAMES_AR.get(disease_key, disease)

        sections = [
            "🌿 **Nabtati (وضع عدم الاتصال) — تحليل الفحص**\n",
            f"📊 **بيانات الفحص:**",
            f"• المرض: **{ar_disease}**",
            f"• الشدة: **{ratio_val:.1f}%** ({stage_ar})",
            f"• الثقة: **{confidence}%**",
            f"\n{urgency}\n",
        ]

        # Get treatments
        treatments = _TREATMENT_RECS.get(disease_key)
        if not treatments:
            for tk, tv in _TREATMENT_RECS.items():
                if disease_key in tk or tk in disease_key:
                    treatments = tv
                    break

        if treatments:
            sections.append("---\n**💊 خطة العلاج:**\n")
            for i, t in enumerate(treatments, 1):
                sections.append(f"{i}. {t}")

        # Arabic severity-specific advice
        sections.append(f"\n---\n**📋 إجراءات حسب الشدة ({severity_ar}):**\n")
        if severity_ar == "حرج":
            sections.extend([
                "• اعزل النباتات المصابة فوراً",
                "• أزل ودمّر الأنسجة المصابة بشدة",
                "• طبق مبيداً فطرياً جهازياً خلال 24 ساعة",
                "• استشر متخصصاً زراعياً بشكل عاجل",
                "• راقب النباتات المجاورة يومياً",
            ])
        elif severity_ar == "متقدم":
            sections.extend([
                "• ابدأ العلاج بمبيد فطري مُستهدف",
                "• حسّن دوران الهواء حول النباتات",
                "• عدّل الري لتقليل رطوبة الأوراق",
                "• حدد موعد فحص متابعة بعد 3 أيام",
            ])
        elif severity_ar == "متوسط":
            sections.extend([
                "• طبق مبيداً فطرياً وقائياً",
                "• راقب النبات يومياً لمتابعة التطور",
                "• تأكد من توفير التغذية المناسبة",
                "• أعد الفحص بعد 5-7 أيام لتتبع التغييرات",
            ])
        else:
            sections.extend([
                "• واصل المراقبة المنتظمة",
                "• طبق مبيداً فطرياً حيوياً وقائياً",
                "• حافظ على ظروف نمو مثالية",
                "• وثّق بفحوصات متابعة",
            ])

        # Answer user's specific question in Arabic
        user_question = message.strip()
        if user_question:
            keywords = self._extract_keywords(message)
            normalized = self._normalize_keywords(keywords)
            nutrients = self._match_nutrients(normalized)

            if nutrients:
                sections.append("\n---\n**🧪 إضافي — تحليل العناصر الغذائية:**\n")
                for nutrient_name, data in nutrients:
                    ar_name = {
                        "nitrogen": "النيتروجين", "phosphorus": "الفوسفور",
                        "potassium": "البوتاسيوم", "calcium": "الكالسيوم",
                        "magnesium": "المغنيسيوم", "iron": "الحديد",
                        "sulfur": "الكبريت",
                    }.get(nutrient_name, nutrient_name)
                    sections.append(f"**{ar_name}:**")
                    advice = data.get("advice_ar", data["advice"])
                    for a in advice[:3]:
                        sections.append(f"• {a}")

        sections.append(
            "\n---\n_📡 تم إنشاء هذه الاستجابة في وضع عدم الاتصال باستخدام بيانات الفحص._"
        )

        return {"text": "\n".join(sections), "status": "ok", "offline": True}


# Singleton instance
_local_ai_instance = None


def get_local_ai_service() -> LocalAIAssistant:
    """Get or create the singleton LocalAIAssistant instance."""
    global _local_ai_instance
    if _local_ai_instance is None:
        _local_ai_instance = LocalAIAssistant()
    return _local_ai_instance
