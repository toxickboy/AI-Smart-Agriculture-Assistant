"""
Agricultural Chatbot
--------------------
Fast rule-based answers for well-known farming topics.
For anything the rule engine can't answer, falls back to the Gemini AI
(via the official google-genai SDK) for a rich, context-aware response.
"""

import os
import re
import random
from typing import Optional

# ── Google Gen AI SDK ──────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ── Knowledge base (fast, offline answers) ────────────────────────────────
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": [r"hi\b", r"hello", r"hey", r"namaskar", r"namaste", r"good morning", r"good evening"],
        "responses": [
            "Hello! I'm your AI Farm Assistant 🌱 How can I help you today? You can ask me about crops, diseases, fertilizers, or weather.",
            "Namaskar! 🙏 Welcome to AI Smart Agriculture Assistant. Ask me anything about farming!",
            "Hi there, farmer friend! 🌾 Ready to help with your agricultural queries.",
        ]
    },
    "crop_season": {
        "patterns": [r"kharif", r"rabi", r"zaid", r"season", r"when to sow", r"sowing time", r"planting time"],
        "responses": [
            "🌱 **Agricultural Seasons in India:**\n\n"
            "**Kharif (June-November):** Rice, Maize, Cotton, Soybean, Groundnut, Sugarcane\n\n"
            "**Rabi (November-March):** Wheat, Chickpea, Mustard, Potato, Pea, Lentil\n\n"
            "**Zaid/Summer (March-June):** Watermelon, Muskmelon, Cucumber, Fodder crops\n\n"
            "Consult your local KVK for region-specific sowing calendars."
        ]
    },
    "fertilizer": {
        "patterns": [r"fertiliz", r"npk", r"urea", r"dap", r"nitrogen", r"phosphorus", r"potassium", r"nutrient", r"khad"],
        "responses": [
            "🌿 **Fertilizer Guide:**\n\n"
            "**N (Nitrogen):** Promotes leafy growth. Source: Urea (46% N), CAN\n"
            "**P (Phosphorus):** Root development & flowering. Source: DAP, SSP\n"
            "**K (Potassium):** Disease resistance & fruit quality. Source: MOP, SOP\n\n"
            "**General NPK ratios:**\n"
            "• Rice: 120:60:60 kg/ha\n"
            "• Wheat: 120:60:40 kg/ha\n"
            "• Maize: 150:75:75 kg/ha\n\n"
            "⚠️ Always do a soil test before heavy fertilizer application!"
        ]
    },
    "irrigation": {
        "patterns": [r"irrigat", r"water", r"drip", r"sprinkler", r"flood", r"pani", r"watering"],
        "responses": [
            "💧 **Irrigation Methods:**\n\n"
            "**Drip Irrigation:** 40-50% water saving. Best for vegetables, orchards.\n"
            "**Sprinkler:** Good for wheat, groundnut, vegetables. 30% saving.\n"
            "**Flood/Furrow:** Suitable for rice, sugarcane. Traditional method.\n\n"
            "**Government subsidy:** 55-75% subsidy on drip/sprinkler under PMKSY scheme.\n\n"
            "**Critical irrigation stages:**\n"
            "• Rice: 5-7 cm standing water during crop period\n"
            "• Wheat: Crown root, tillering, flowering, grain fill\n"
            "• Maize: Knee-high, tasseling, grain fill stages"
        ]
    },
    "pest_disease": {
        "patterns": [r"pest", r"insect", r"disease", r"blight", r"rust", r"fungus", r"virus", r"bacteria", r"rot", r"wilt", r"kida", r"bug"],
        "responses": [
            "🐛 **Integrated Pest Management (IPM):**\n\n"
            "**Prevention first:**\n"
            "• Use certified disease-free seeds\n"
            "• Crop rotation to break pest cycles\n"
            "• Maintain field hygiene - remove crop debris\n\n"
            "**Biological control:**\n"
            "• Neem-based sprays (3-5%)\n"
            "• Trichoderma for soil-borne diseases\n"
            "• Pheromone traps for moths\n\n"
            "**Chemical control (last resort):**\n"
            "• Always read pesticide label carefully\n"
            "• Follow PHI (Pre-Harvest Interval)\n"
            "• Wear full PPE during application\n\n"
            "Use our **Disease Detection** tab to identify plant diseases from photos!"
        ]
    },
    "soil_health": {
        "patterns": [r"soil", r"ph", r"maati", r"mati", r"soil test", r"soil health", r"compost", r"organic"],
        "responses": [
            "🌍 **Soil Health Management:**\n\n"
            "**Ideal pH ranges:**\n"
            "• Most crops: 6.0 - 7.5\n"
            "• Rice: 5.5 - 6.5\n"
            "• Blueberries: 4.5 - 5.5\n\n"
            "**Improving acidic soil (pH < 6):** Apply agricultural lime @ 2-4 tonnes/ha\n"
            "**Improving alkaline soil (pH > 7.5):** Apply gypsum or sulfur\n\n"
            "**Organic Matter:**\n"
            "• Apply FYM @ 10-15 tonnes/ha\n"
            "• Green manuring with dhaincha or sesbania\n"
            "• Vermicompost 2-3 tonnes/ha\n\n"
            "**Soil testing:** Visit nearest soil testing lab or use Soil Health Card scheme"
        ]
    },
    "organic_farming": {
        "patterns": [r"organic", r"natural farm", r"chemical free", r"bio", r"jeevamrit", r"panchagavya"],
        "responses": [
            "🌿 **Organic Farming Guide:**\n\n"
            "**Biofertilizers:**\n"
            "• Rhizobium for legumes\n"
            "• Azospirillum for cereals\n"
            "• PSB (Phosphate Solubilizing Bacteria)\n"
            "• Mycorrhiza for root development\n\n"
            "**Natural pesticides:**\n"
            "• Neem oil (5 ml/L) - broad spectrum\n"
            "• Garlic-chili spray - sucking pests\n"
            "• Panchagavya (3%) - growth promoter\n"
            "• Jeevamrit - soil microbiome booster\n\n"
            "**Certification:** Apply under Participatory Guarantee System (PGS-India)\n"
            "**Subsidy:** Check PKVY scheme for organic cluster farming incentives"
        ]
    },
    "government_schemes": {
        "patterns": [r"scheme", r"subsid", r"government", r"yojana", r"help", r"loan", r"insurance", r"pm kisan", r"pmfby"],
        "responses": [
            "🏛️ **Key Government Schemes for Farmers:**\n\n"
            "**PM-KISAN:** ₹6,000/year direct income support (3 installments)\n\n"
            "**PMFBY (Crop Insurance):** Subsidized crop insurance against natural calamities\n\n"
            "**KCC (Kisan Credit Card):** Short-term crop loans @ 4-7% interest\n\n"
            "**PMKSY:** 55-75% subsidy on drip/sprinkler irrigation systems\n\n"
            "**eNAM:** Online agricultural marketplace for better prices\n\n"
            "**Soil Health Card:** Free soil testing every 3 years\n\n"
            "📞 Contact your nearest **Krishi Vigyan Kendra (KVK)** or **Agriculture Department** for details."
        ]
    },
    "crop_recommendation": {
        "patterns": [r"which crop", r"best crop", r"what to grow", r"what crop", r"crop suggest", r"crop recom"],
        "responses": [
            "🌱 For personalized crop recommendations, please use our **Crop Recommendation** tab!\n\n"
            "Enter your soil parameters (N, P, K, pH) and climate data to get AI-powered crop suggestions.\n\n"
            "**Quick guide by soil type:**\n"
            "• Black soil (Vertisol): Cotton, Sorghum, Wheat, Soybean\n"
            "• Red soil: Groundnut, Ragi, Pulses, Millets\n"
            "• Alluvial soil: Rice, Wheat, Sugarcane, Vegetables\n"
            "• Laterite soil: Tea, Coffee, Cashew, Tapioca\n"
            "• Sandy soil: Groundnut, Potato, Carrot, Watermelon"
        ]
    },
    "disease_detection": {
        "patterns": [r"disease detect", r"leaf disease", r"plant disease", r"identify disease", r"diagnos"],
        "responses": [
            "🔬 For plant disease detection, use our **Disease Detection** tab!\n\n"
            "📸 Simply upload a clear photo of the affected leaf and our AI will:\n"
            "• Identify the disease name\n"
            "• Explain symptoms\n"
            "• Recommend pesticides with dosage\n"
            "• Suggest cultural practices\n\n"
            "**Tips for best results:**\n"
            "• Use natural daylight for photos\n"
            "• Include both healthy and affected areas in the image\n"
            "• Take close-up shots of disease symptoms"
        ]
    },
    "market_price": {
        "patterns": [r"price", r"market", r"msp", r"sell", r"mandi", r"rate", r"bhav"],
        "responses": [
            "💰 **Agricultural Market Information:**\n\n"
            "**MSP (Minimum Support Price) 2024-25 (Approximate):**\n"
            "• Paddy (Common): ₹2,300/quintal\n"
            "• Wheat: ₹2,275/quintal\n"
            "• Maize: ₹2,090/quintal\n"
            "• Soybean: ₹4,892/quintal\n"
            "• Cotton (Medium): ₹7,121/quintal\n\n"
            "**For live mandi prices:**\n"
            "📱 Kisan Suvidha App\n"
            "🌐 agmarknet.gov.in\n"
            "📱 eNAM app (National Agriculture Market)\n\n"
            "Note: Prices vary by location and quality."
        ]
    },
    "weather_query": {
        "patterns": [r"weather", r"rain", r"temperature", r"forecast", r"hawa", r"mausam", r"barish"],
        "responses": [
            "🌤️ For current weather and farming advice, check our **Weather** tab!\n\n"
            "You can:\n"
            "• Get weather for any Indian city\n"
            "• See temperature, humidity, wind speed\n"
            "• Get personalized farming advisories\n"
            "• View seasonal crop recommendations\n\n"
            "**Weather alerts to watch:**\n"
            "• Temperature > 40°C: Increase irrigation frequency\n"
            "• Humidity > 80%: Spray preventive fungicide\n"
            "• Wind > 20 km/h: Avoid pesticide application\n"
            "• Rain forecast: Postpone fertilizer application"
        ]
    },
    "thanks": {
        "patterns": [r"thank", r"thanks", r"dhanyawad", r"shukriya", r"helpful", r"great", r"good"],
        "responses": [
            "You're welcome! Happy farming! 🌾 Feel free to ask anytime.",
            "Glad I could help! 🌱 Wishing you a great harvest!",
            "Anytime! 🙏 Remember to always consult your local KVK for expert guidance.",
        ]
    },
    "farewell": {
        "patterns": [r"bye", r"goodbye", r"see you", r"quit", r"exit", r"alvida"],
        "responses": [
            "Goodbye! Happy Farming! 🌾🙏",
            "Alvida! May your crops be plentiful! 🌱",
            "See you soon! Keep farming smart! 🚜",
        ]
    },
    "help": {
        "patterns": [r"help", r"what can you do", r"features", r"commands", r"guide"],
        "responses": [
            "🤖 **I can help you with:**\n\n"
            "🌱 Crop recommendations and seasons\n"
            "🐛 Pest and disease management\n"
            "💧 Irrigation methods and scheduling\n"
            "🌿 Fertilizer guidance (NPK)\n"
            "🌍 Soil health and testing\n"
            "🌿 Organic farming methods\n"
            "🏛️ Government schemes and subsidies\n"
            "💰 Market prices and MSP\n"
            "🌤️ Weather and farming advisories\n\n"
            "Also use our specialized tools:\n"
            "• **Crop Recommendation** - AI crop prediction\n"
            "• **Disease Detection** - Upload leaf photo\n"
            "• **Weather** - City-wise weather & advice"
        ]
    },
}

FALLBACK_RESPONSES = [
    "I'm not sure about that specific query. Could you rephrase or ask about crops, diseases, fertilizers, irrigation, or government schemes? 🌱",
    "That's an interesting question! For specific local guidance, I recommend contacting your nearest Krishi Vigyan Kendra (KVK). 📞",
    "I don't have information on that topic yet. Try asking about: crop seasons, fertilizers, irrigation, pest management, or government schemes. 🤔",
    "Good question! Please consult your local agriculture extension officer for region-specific advice. Meanwhile, I can help with general farming queries! 🙏",
]

# ── Gemini AI fallback ─────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = (
    "You are AgriBot, a knowledgeable and friendly AI assistant for Indian farmers. "
    "Answer agriculture-related questions clearly and practically. "
    "Focus on crops, soil, fertilizers, pest management, irrigation, market prices, "
    "and government schemes relevant to India. "
    "Keep answers concise (under 200 words) and use simple language. "
    "If a question is unrelated to agriculture, politely redirect to farming topics."
)

_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def _get_gemini_response(user_input: str, api_key: str) -> Optional[str]:
    """
    Call Gemini with the official SDK and return the response text.
    Returns None on any failure so the caller can fall back gracefully.
    """
    if not _GENAI_AVAILABLE or not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)

        last_error: Exception | None = None
        for model_name in _GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_input,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_SYSTEM_INSTRUCTION,
                        temperature=0.7,
                        max_output_tokens=350,
                    ),
                )
                text = (response.text or "").strip()
                return text if text else None

            except Exception as exc:  # noqa: BLE001
                err_str = str(exc).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    last_error = exc
                    continue
                # Non-availability error → surface as None so rule-base wins.
                return None

    except Exception:  # noqa: BLE001
        return None

    return None  # All models exhausted


# ── Public API ─────────────────────────────────────────────────────────────

def get_chatbot_response(user_input: str, language: str = "en") -> str:
    """
    Return a chatbot response.

    Priority:
      1. Fast rule-based match from KNOWLEDGE_BASE
      2. Crop-specific static info
      3. Gemini AI (if GEMINI_API_KEY is set in the environment)
      4. Static fallback message
    """
    user_lower = user_input.lower().strip()

    # ── 1. Knowledge base ──────────────────────────────────────────────────
    for category, data in KNOWLEDGE_BASE.items():
        for pattern in data["patterns"]:
            if re.search(pattern, user_lower):
                return random.choice(data["responses"])

    # ── 2. Quantity / measurement queries ─────────────────────────────────
    if re.search(r"\d+", user_input) and re.search(r"(acre|hectare|kg|litre)", user_lower):
        return (
            "📊 For specific quantity calculations, I recommend using our Crop Recommendation tool "
            "with your exact field measurements. For precise fertilizer dosage calculations, "
            "consult your local agriculture officer who can consider your specific soil test results."
        )

    # ── 3. Crop-specific static info ───────────────────────────────────────
    crops = [
        "rice", "wheat", "maize", "cotton", "sugarcane", "soybean",
        "onion", "tomato", "potato", "groundnut", "paddy", "bajra", "jowar", "ragi",
    ]
    for crop in crops:
        if crop in user_lower:
            return get_crop_info(crop)

    # ── 4. Gemini AI fallback ──────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key and _GENAI_AVAILABLE:
        gemini_answer = _get_gemini_response(user_input, api_key)
        if gemini_answer:
            return f"🤖 *(AI Answer)*\n\n{gemini_answer}"

    # ── 5. Static fallback ─────────────────────────────────────────────────
    return random.choice(FALLBACK_RESPONSES)


# ── Crop detail helper ────────────────────────────────────────────────────

def get_crop_info(crop: str) -> str:
    """Return a static farming guide for common crops."""
    crop_details = {
        "rice": {
            "season": "Kharif (June-Nov)", "duration": "120-150 days",
            "water": "High (1200-2000 mm)", "temp": "20-35°C", "ph": "5.5-6.5",
            "fertilizer": "N:P:K = 120:60:60 kg/ha",
            "tip": "Transplanting at 21-25 days old seedlings gives better yield.",
        },
        "wheat": {
            "season": "Rabi (Nov-Mar)", "duration": "110-130 days",
            "water": "Medium (450-650 mm)", "temp": "15-25°C", "ph": "6.0-7.5",
            "fertilizer": "N:P:K = 120:60:40 kg/ha",
            "tip": "Timely sowing (Nov 15 - Dec 15) is critical for yield in North India.",
        },
        "maize": {
            "season": "Kharif/Rabi", "duration": "90-120 days",
            "water": "Medium (600-900 mm)", "temp": "21-27°C", "ph": "5.8-7.0",
            "fertilizer": "N:P:K = 150:75:75 kg/ha",
            "tip": "Tasseling stage is most critical - ensure no water stress.",
        },
        "cotton": {
            "season": "Kharif (May-Nov)", "duration": "150-180 days",
            "water": "Medium (700-1200 mm)", "temp": "21-35°C", "ph": "5.8-8.0",
            "fertilizer": "N:P:K = 120:60:60 kg/ha (Bt cotton)",
            "tip": "Use Bt cotton hybrids for bollworm resistance.",
        },
        "tomato": {
            "season": "Year-round", "duration": "60-90 days",
            "water": "Medium-High (400-600 mm)", "temp": "18-27°C", "ph": "6.0-7.0",
            "fertilizer": "N:P:K = 120:80:80 kg/ha",
            "tip": "Staking and pruning suckers improves yield and reduces disease.",
        },
        "potato": {
            "season": "Rabi (Oct-Feb)", "duration": "75-120 days",
            "water": "Medium (500-700 mm)", "temp": "15-25°C", "ph": "5.0-6.0",
            "fertilizer": "N:P:K = 180:100:100 kg/ha",
            "tip": "Use certified seed tubers to avoid virus diseases.",
        },
        "onion": {
            "season": "Rabi/Kharif", "duration": "120-150 days",
            "water": "Medium (350-550 mm)", "temp": "13-24°C", "ph": "6.0-7.5",
            "fertilizer": "N:P:K = 100:50:50 kg/ha",
            "tip": "Stop irrigation 10 days before harvest for better storage.",
        },
    }

    if crop in crop_details:
        info = crop_details[crop]
        return (
            f"🌱 **{crop.upper()} Farming Guide:**\n\n"
            f"📅 **Season:** {info['season']}\n"
            f"⏱️ **Duration:** {info['duration']}\n"
            f"💧 **Water Need:** {info['water']}\n"
            f"🌡️ **Temperature:** {info['temp']}\n"
            f"🌍 **Soil pH:** {info['ph']}\n"
            f"🌿 **Fertilizer:** {info['fertilizer']}\n\n"
            f"💡 **Pro Tip:** {info['tip']}\n\n"
            f"Use our **Crop Recommendation** tool for AI-powered suggestions!"
        )

    return (
        f"I have limited specific information on {crop}. "
        f"Please use our **Crop Recommendation** tool with your soil parameters, "
        f"or consult your local KVK for detailed guidance on {crop} cultivation."
    )
