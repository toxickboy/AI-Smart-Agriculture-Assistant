"""
Weather Module
Uses OpenWeatherMap free tier API or mock weather system
"""

import json
import random
import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Free OpenWeatherMap API key (user should replace with own free key)
# Get free key at: https://openweathermap.org/api
OWM_API_KEY = "demo"  # Replace with actual free key
OWM_BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

MOCK_CITIES = {
    "mumbai": {"temp": 32, "humidity": 78, "wind": 12, "desc": "Partly Cloudy", "pressure": 1010},
    "pune": {"temp": 28, "humidity": 62, "wind": 8, "desc": "Clear Sky", "pressure": 1013},
    "nagpur": {"temp": 35, "humidity": 45, "wind": 10, "desc": "Sunny", "pressure": 1008},
    "nashik": {"temp": 27, "humidity": 58, "wind": 7, "desc": "Overcast", "pressure": 1012},
    "aurangabad": {"temp": 30, "humidity": 50, "wind": 9, "desc": "Partly Cloudy", "pressure": 1011},
    "delhi": {"temp": 38, "humidity": 40, "wind": 15, "desc": "Hazy", "pressure": 1005},
    "bangalore": {"temp": 26, "humidity": 72, "wind": 6, "desc": "Light Rain", "pressure": 1014},
    "hyderabad": {"temp": 33, "humidity": 55, "wind": 11, "desc": "Clear Sky", "pressure": 1009},
    "chennai": {"temp": 35, "humidity": 80, "wind": 14, "desc": "Humid", "pressure": 1007},
    "kolkata": {"temp": 34, "humidity": 82, "wind": 10, "desc": "Thunderstorm", "pressure": 1004},
    "ahmedabad": {"temp": 36, "humidity": 42, "wind": 12, "desc": "Hot & Sunny", "pressure": 1006},
    "jaipur": {"temp": 37, "humidity": 38, "wind": 13, "desc": "Sunny", "pressure": 1007},
    "lucknow": {"temp": 36, "humidity": 48, "wind": 9, "desc": "Partly Cloudy", "pressure": 1008},
    "patna": {"temp": 34, "humidity": 68, "wind": 8, "desc": "Humid", "pressure": 1009},
    "bhopal": {"temp": 32, "humidity": 52, "wind": 10, "desc": "Clear Sky", "pressure": 1011},
    "indore": {"temp": 31, "humidity": 54, "wind": 9, "desc": "Partly Cloudy", "pressure": 1012},
    "amravati": {"temp": 34, "humidity": 47, "wind": 8, "desc": "Sunny", "pressure": 1010},
    "solapur": {"temp": 33, "humidity": 44, "wind": 10, "desc": "Hot & Dry", "pressure": 1009},
    "kolhapur": {"temp": 29, "humidity": 65, "wind": 7, "desc": "Overcast", "pressure": 1013},
    "default": {"temp": 30, "humidity": 60, "wind": 10, "desc": "Partly Cloudy", "pressure": 1010},
}


def get_farming_advice(temp, humidity, wind_speed, description):
    """Generate farming advice based on weather conditions"""
    advice = []
    alerts = []

    # Temperature advice
    if temp > 40:
        alerts.append("🔥 Extreme heat! Irrigate crops early morning or evening only.")
        advice.append("Avoid field work during peak afternoon hours (12-3 PM).")
    elif temp > 35:
        advice.append("🌡️ High temperature – ensure adequate irrigation, especially for vegetables.")
    elif temp < 10:
        alerts.append("❄️ Low temperature – protect crops from frost. Cover seedlings overnight.")
    elif 20 <= temp <= 30:
        advice.append("✅ Ideal temperature range for most crops.")

    # Humidity advice
    if humidity > 80:
        alerts.append("⚠️ High humidity – increased risk of fungal diseases. Monitor crops closely.")
        advice.append("Spray preventive fungicide if humidity persists above 80% for 3+ days.")
    elif humidity < 30:
        advice.append("💧 Low humidity – increase irrigation frequency for moisture-loving crops.")
        advice.append("Mulching recommended to reduce soil moisture evaporation.")
    elif 50 <= humidity <= 70:
        advice.append("✅ Humidity levels are suitable for most farming activities.")

    # Wind advice
    if wind_speed > 20:
        alerts.append("💨 Strong winds – avoid pesticide spraying today to prevent drift.")
        advice.append("Stake tall crops like maize and sugarcane to prevent lodging.")
    elif wind_speed < 5:
        advice.append("Good conditions for pesticide application (low wind drift).")

    # Description-based advice
    if "rain" in description.lower() or "thunderstorm" in description.lower():
        alerts.append("🌧️ Rain expected – postpone fertilizer and pesticide application.")
        advice.append("Check field drainage to prevent waterlogging.")
        advice.append("Ideal time for dry land sowing after rain stops.")
    elif "clear" in description.lower() or "sunny" in description.lower():
        advice.append("☀️ Good day for field operations, harvesting, and crop drying.")

    # Combined conditions
    if temp > 28 and humidity > 70:
        advice.append("🍄 Warm & humid conditions favour disease development. Scout fields daily.")

    if not advice and not alerts:
        advice.append("Weather conditions appear normal for farming activities.")

    return advice, alerts


def get_weather_owm(city: str, api_key: str):
    """Fetch weather from OpenWeatherMap API"""
    if not REQUESTS_AVAILABLE:
        return None

    try:
        url = f"{OWM_BASE_URL}?q={city}&appid={api_key}&units=metric"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                'city': data.get('name', city),
                'country': data['sys'].get('country', 'IN'),
                'temperature': round(data['main']['temp'], 1),
                'feels_like': round(data['main']['feels_like'], 1),
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': round(data['wind']['speed'] * 3.6, 1),  # m/s to km/h
                'description': data['weather'][0]['description'].title(),
                'icon': data['weather'][0]['icon'],
                'visibility': data.get('visibility', 10000) / 1000,
                'source': 'OpenWeatherMap API',
            }
    except Exception:
        pass
    return None


def get_weather_mock(city: str):
    """Return mock weather data"""
    city_lower = city.lower().strip()
    base = MOCK_CITIES.get(city_lower, MOCK_CITIES['default'])

    # Add slight variation
    variation = random.uniform(-2, 2)
    return {
        'city': city.title(),
        'country': 'IN',
        'temperature': round(base['temp'] + variation, 1),
        'feels_like': round(base['temp'] + variation + 2, 1),
        'humidity': base['humidity'] + random.randint(-5, 5),
        'pressure': base['pressure'],
        'wind_speed': base['wind'] + round(random.uniform(-2, 2), 1),
        'description': base['desc'],
        'icon': '02d',
        'visibility': round(random.uniform(8, 12), 1),
        'source': 'Demo Mode (Mock Weather)',
    }


def get_weather(city: str, api_key: str = None):
    """Main weather function - tries API, falls back to mock"""
    weather = None

    if api_key and api_key != "demo" and REQUESTS_AVAILABLE:
        weather = get_weather_owm(city, api_key)

    if not weather:
        weather = get_weather_mock(city)

    # Add farming advice
    advice, alerts = get_farming_advice(
        weather['temperature'],
        weather['humidity'],
        weather['wind_speed'],
        weather['description']
    )
    weather['farming_advice'] = advice
    weather['alerts'] = alerts
    weather['timestamp'] = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

    return weather


def get_seasonal_crops(month: int = None):
    """Return crops suitable for current/given month"""
    if month is None:
        month = datetime.datetime.now().month

    seasonal = {
        # Kharif season (June-November)
        6: ['Rice', 'Maize', 'Cotton', 'Soybean', 'Groundnut'],
        7: ['Rice', 'Maize', 'Cotton', 'Pigeonpeas', 'Mungbean'],
        8: ['Rice', 'Maize', 'Cotton', 'Blackgram', 'Sorghum'],
        9: ['Rice', 'Maize', 'Sugarcane', 'Turmeric'],
        10: ['Rice (harvest)', 'Maize (harvest)', 'Rabi sowing preparation'],
        11: ['Wheat', 'Chickpea', 'Mustard', 'Potato', 'Onion'],
        # Rabi season (November-March)
        12: ['Wheat', 'Chickpea', 'Lentil', 'Mustard', 'Sunflower'],
        1: ['Wheat', 'Chickpea', 'Lentil', 'Potato', 'Pea'],
        2: ['Wheat', 'Chickpea', 'Lentil', 'Sunflower'],
        3: ['Wheat (harvest)', 'Chickpea (harvest)', 'Summer crops sowing'],
        # Zaid/Summer (April-June)
        4: ['Watermelon', 'Muskmelon', 'Cucumber', 'Fodder crops'],
        5: ['Watermelon', 'Muskmelon', 'Moong (summer)', 'Sunflower'],
    }
    return seasonal.get(month, ['Consult local agriculture office'])
