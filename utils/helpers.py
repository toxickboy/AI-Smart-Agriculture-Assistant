"""
Helper utilities for the Smart Agriculture Assistant
"""

import json
import os
import base64
from pathlib import Path

TRANSLATIONS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'translations.json')


def load_translations():
    """Load translation data"""
    try:
        with open(TRANSLATIONS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"en": {}, "mr": {}}


def get_text(key: str, language: str = "en") -> str:
    """Get translated text"""
    translations = load_translations()
    lang_dict = translations.get(language, translations.get("en", {}))
    return lang_dict.get(key, translations.get("en", {}).get(key, key))


def get_severity_color(severity: str) -> str:
    """Return color based on severity level"""
    severity_lower = severity.lower()
    if "very high" in severity_lower or "emergency" in severity_lower:
        return "#FF0000"
    elif "high" in severity_lower:
        return "#FF6B00"
    elif "moderate" in severity_lower:
        return "#FFC107"
    elif "low" in severity_lower or "none" in severity_lower:
        return "#28A745"
    return "#6C757D"


def get_ph_status(ph: float) -> dict:
    """Return pH interpretation"""
    if ph < 4.5:
        return {"status": "Strongly Acidic", "color": "#DC3545", "emoji": "⚠️",
                "advice": "Apply heavy lime application. Unsuitable for most crops."}
    elif ph < 5.5:
        return {"status": "Acidic", "color": "#FD7E14", "emoji": "⚠️",
                "advice": "Apply agricultural lime @ 2-4 tonnes/ha to raise pH."}
    elif ph < 6.5:
        return {"status": "Slightly Acidic - Good", "color": "#28A745", "emoji": "✅",
                "advice": "Ideal range for most crops including rice, vegetables."}
    elif ph < 7.5:
        return {"status": "Neutral - Ideal", "color": "#20C997", "emoji": "✅",
                "advice": "Perfect for wheat, maize, legumes and most vegetables."}
    elif ph < 8.5:
        return {"status": "Alkaline", "color": "#FD7E14", "emoji": "⚠️",
                "advice": "Apply gypsum or sulfur to lower pH. Suitable for few crops."}
    else:
        return {"status": "Strongly Alkaline", "color": "#DC3545", "emoji": "⛔",
                "advice": "Heavy soil amendment needed. Very few crops can survive."}


def get_npk_status(N: float, P: float, K: float) -> dict:
    """Return NPK level interpretation"""
    def level(val, low, high):
        if val < low:
            return "Low", "#FD7E14"
        elif val > high:
            return "Excess", "#DC3545"
        return "Optimal", "#28A745"

    n_status, n_color = level(N, 40, 120)
    p_status, p_color = level(P, 20, 80)
    k_status, k_color = level(K, 20, 80)

    return {
        "N": {"value": N, "status": n_status, "color": n_color,
              "advice": "Apply Urea" if n_status == "Low" else ("Reduce N fertilizer" if n_status == "Excess" else "N is adequate")},
        "P": {"value": P, "status": p_status, "color": p_color,
              "advice": "Apply DAP/SSP" if p_status == "Low" else ("Reduce P fertilizer" if p_status == "Excess" else "P is adequate")},
        "K": {"value": K, "status": k_status, "color": k_color,
              "advice": "Apply MOP/SOP" if k_status == "Low" else ("Reduce K fertilizer" if k_status == "Excess" else "K is adequate")},
    }


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def get_weather_icon_emoji(description: str) -> str:
    """Convert weather description to emoji"""
    desc = description.lower()
    if "thunder" in desc:
        return "⛈️"
    elif "rain" in desc:
        return "🌧️"
    elif "drizzle" in desc:
        return "🌦️"
    elif "snow" in desc:
        return "❄️"
    elif "mist" in desc or "fog" in desc or "haze" in desc:
        return "🌫️"
    elif "cloud" in desc:
        return "⛅"
    elif "clear" in desc or "sunny" in desc:
        return "☀️"
    elif "overcast" in desc:
        return "☁️"
    elif "hot" in desc:
        return "🌡️"
    return "🌤️"


def format_confidence(value: float) -> str:
    """Format confidence percentage"""
    pct = value * 100
    if pct >= 80:
        return f"🟢 {pct:.1f}%"
    elif pct >= 60:
        return f"🟡 {pct:.1f}%"
    else:
        return f"🔴 {pct:.1f}%"


def get_crop_emoji(crop_name: str) -> str:
    """Get emoji for crop name"""
    emoji_map = {
        'rice': '🌾', 'wheat': '🌾', 'maize': '🌽', 'corn': '🌽',
        'chickpea': '🫘', 'kidneybeans': '🫘', 'pigeonpeas': '🫘',
        'mungbean': '🫘', 'blackgram': '🫘', 'lentil': '🫘',
        'mothbeans': '🫘', 'banana': '🍌', 'mango': '🥭',
        'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈',
        'apple': '🍏', 'orange': '🍊', 'papaya': '🧡',
        'coconut': '🥥', 'cotton': '🌿', 'jute': '🌿',
        'coffee': '☕', 'pomegranate': '🍎', 'sugarcane': '🎋',
        'tomato': '🍅', 'potato': '🥔', 'onion': '🧅',
        'soybean': '🫘', 'groundnut': '🥜', 'sunflower': '🌻',
        'default': '🌱',
    }
    return emoji_map.get(crop_name.lower(), emoji_map['default'])


def load_medicine_db():
    """Load medicine database"""
    medicine_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data', 'medicine_db.json'
    )
    with open(medicine_path, 'r', encoding='utf-8') as f:
        return json.load(f)
