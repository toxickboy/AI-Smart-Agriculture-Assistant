"""
AI Smart Agriculture Assistant
Main Streamlit Application
Run with: streamlit run app.py
"""

import os
import sys
import json
import datetime
import streamlit as st
from PIL import Image


# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def load_env_file(env_path: str = ".env"):
    """Minimal .env loader to keep API-key flow simple and dependency-free."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


load_env_file()

from models.crop_model import predict_crop, get_soil_health, load_model
from utils.gemini_disease import analyze_disease_image_with_gemini
from utils.weather import get_weather, get_seasonal_crops
from utils.chatbot import get_chatbot_response
from utils.helpers import (
    get_text, get_ph_status, get_npk_status,
    get_weather_icon_emoji, format_confidence, get_crop_emoji
)

# â”€â”€ Page Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="AI Smart Agriculture Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: Green;
    }

    /* â”€â”€ MAIN AREA background â”€â”€ */
    .stApp { background-color: White; }

    /* â”€â”€ HEADER â”€â”€ */
    .main-header {
        background: linear-gradient(135deg, #145A32 0%, #1E8449 100%);
        color: #FFFFFF;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 24px rgba(20,90,50,0.35);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #FFFFFF; }
    .main-header p  { font-size: 0.97rem; margin: 0.4rem 0 0; color: #D5F5E3; }

    /* â”€â”€ SECTION HEADER â”€â”€ */
    .section-header {
        color: #145A32;
        border-bottom: 3px solid #27AE60;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    /* â”€â”€ CARDS â”€â”€ */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.10);
        border-left: 5px solid #1E8449;
        margin-bottom: 1rem;
        color: #111111;
    }
    .metric-card h3 { color: #145A32; font-size: 1.4rem; margin: 0; }
    .metric-card p  { color: #333333; margin: 0.3rem 0 0; font-size: 0.9rem; }

    /* â”€â”€ RESULT BOXES â”€â”€ */
    .result-box {
        background: #EAFAF1;
        border: 2px solid #27AE60;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0B3D1F;
    }
    .result-box h2 { color: #145A32; }

    .disease-box {
        background: #FEF9E7;
        border: 2px solid #D4AC0D;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #4D3800;
    }
    .disease-box h3 { color: #7D5C00; }

    .healthy-box {
        background: #EAFAF1;
        border: 2px solid #1E8449;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0B3D1F;
    }
    .healthy-box h3 { color: #145A32; }

    .danger-box {
        background: #FDEDEC;
        border: 2px solid #C0392B;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        color: #6E0E0A;
        font-weight: 500;
    }

    .warning-box {
        background: #FEF9E7;
        border: 2px solid #B7950B;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 0.6rem 0;
        color: #4D3800;
        font-weight: 500;
    }

    .info-box {
        background: #EBF5FB;
        border: 2px solid #2E86C1;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #1A3C54;
        font-size: 0.9rem;
        line-height: 1.7;
    }
    .info-box b { color: #1A5276; }
    .info-box a { color: #1A5276; }

    .disclaimer-box {
        background: #FEF5E7;
        border: 2px solid #CA6F1E;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1.5rem 0 0.5rem;
        font-size: 0.87rem;
        color: #7E3200;
        font-weight: 600;
        line-height: 1.6;
    }

    /* â”€â”€ MEDICINE CARD â”€â”€ */
    .medicine-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.09);
        border-top: 4px solid #1E8449;
        color: #111111;
    }
    .medicine-card b { color: #145A32; }
    .medicine-card table { color: #222222; }

    /* â”€â”€ CHAT MESSAGES â”€â”€ */
    .chat-message-user {
        background: #D6EAF8;
        border-radius: 14px 14px 4px 14px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        text-align: right;
        color: #1A3C54;
        font-weight: 500;
        border: 1px solid #85C1E9;
    }
    .chat-message-bot {
        background: #EAFAF1;
        border-radius: 14px 14px 14px 4px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        color: #0B3D1F;
        border: 1px solid #82E0AA;
        line-height: 1.7;
    }

    /* â”€â”€ CROP CARD â”€â”€ */
    .crop-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
        text-align: left;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1E8449;
        margin-bottom: 0.8rem;
        color: #111111;
    }
    .crop-card b { color: #145A32; font-size: 1.05rem; }
    .crop-card small { color: #555555; }

    /* â”€â”€ WEATHER CARD â”€â”€ */
    .weather-card {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(13,71,161,0.4);
    }
    .weather-card h2 { color: #FFFFFF; margin: 0.4rem 0; }
    .weather-card p  { color: #BBDEFB; margin: 0.2rem 0; }

    /* â”€â”€ BUTTONS â”€â”€ */
    .stButton > button {
        background: #1E8449 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.4rem !important;
        transition: background 0.2s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        background: #145A32 !important;
        box-shadow: 0 4px 14px rgba(30,132,73,0.45) !important;
    }

    /* â”€â”€ SIDEBAR â”€â”€ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B3D1F 0%, #145A32 100%) !important;
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stRadio label { color: #D5F5E3 !important; font-weight: 500; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #FFFFFF !important; font-weight: 700; }
    [data-testid="stSidebar"] .stSelectbox label { color: #A9DFBF !important; }

    /* â”€â”€ STREAMLIT NATIVE ELEMENTS â”€â”€ */
    .stSlider label, .stTextInput label,
    .stFileUploader label { color: #111111 !important; font-weight: 500; }
    .stMarkdown p { color: #222222; }
    .stExpander { border: 1px solid #AED6B8; border-radius: 10px; }
    div[data-testid="stMetric"] label { color: #333333 !important; }
    div[data-testid="stMetric"] div  { color: #111111 !important; font-weight: 700; }

    /* â”€â”€ BADGE â”€â”€ */
    .badge {
        display: inline-block;
        padding: 0.2em 0.75em;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 20px;
        margin: 0.2rem;
    }

    /* â”€â”€ FORCE ALL TEXT DARK â”€â”€ */
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] td,
    div[data-testid="stMarkdownContainer"] th,
    div[data-testid="stMarkdownContainer"] span { color: #111111 !important; }
    div[data-testid="stMarkdownContainer"] a { color: #1565C0 !important; font-weight: 600; }
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] b { color: #145A32 !important; }
    div[data-testid="stMarkdownContainer"] code {
        background: #F0F0F0 !important; color: #111111 !important;
        padding: 2px 6px; border-radius: 4px;
    }
    div[data-testid="stMarkdownContainer"] pre {
        background: #1E1E1E !important; color: #E8F5E9 !important;
        padding: 1rem; border-radius: 8px;
    }
    .stMarkdown table { width: 100%; border-collapse: collapse; }
    .stMarkdown th {
        background: #EAFAF1 !important; color: #0B3D1F !important;
        padding: 8px 12px; border: 1px solid #A9DFBF; font-weight: 700;
    }
    .stMarkdown td {
        color: #111111 !important; padding: 7px 12px; border: 1px solid #D5E8D4;
    }
    .stMarkdown tr:nth-child(even) td { background: #F7FBF7; }
    .streamlit-expanderContent p,
    .streamlit-expanderContent li { color: #111111 !important; }
    div[data-testid="stMetricValue"] { color: #111111 !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #333333 !important; }
</style>
""", unsafe_allow_html=True)


# â”€â”€ Session State Init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "crop_model_loaded" not in st.session_state:
    st.session_state.crop_model_loaded = False


def t(key): return get_text(key, st.session_state.language)


# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("## AgriAI")
    st.markdown("---")

    st.selectbox("Language", options=["English"], index=0)
    st.session_state.language = "en"

    st.markdown("---")
    st.markdown("<h3 style='color:white'>Navigation</h3>", unsafe_allow_html=True)

    page = st.radio(
        "Go to:",
        options=["Crop Recommendation", "Disease Detection", "Weather & Advisory",
                 "Farm Chatbot", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#FFFFFF; font-size:0.82rem; padding:0.6rem 0.5rem;
                background:rgba(255,255,255,0.10); border-radius:8px; line-height:1.9'>
    <b style="color:#A9DFBF">Help Lines:</b><br>
    Kisan: <b>1800-180-1551</b><br>
    KVK: <b>1800-425-1122</b><br>
    SMS Weather: <b>7738299899</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#D5F5E3; font-size:0.76rem; padding:0.5rem; margin-top:0.8rem; line-height:1.9'>
    <b style="color:#FFFFFF">Powered by:</b><br>
    RandomForest ML · Gemini API · OpenWeatherMap
    </div>
    """, unsafe_allow_html=True)


# â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown(f"""
<div class="main-header">
    <h1>{t('app_title')}</h1>
    <p>{t('app_subtitle')} | Made for Indian Farmers</p>
</div>
""", unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 1: CROP RECOMMENDATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "Crop Recommendation":
    st.markdown(f"<h2 class='section-header'>{t('crop_title')}</h2>", unsafe_allow_html=True)
    st.markdown(f"*{t('crop_desc')}*")

    with st.form("crop_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Soil Nutrients")
            N = st.slider(t('nitrogen'), 0, 140, 90,
                         help="Nitrogen content in soil (kg/ha). Typical range: 40-120")
            P = st.slider(t('phosphorus'), 5, 145, 42,
                         help="Phosphorus content in soil (kg/ha). Typical range: 20-80")
            K = st.slider(t('potassium'), 5, 205, 43,
                         help="Potassium content in soil (kg/ha). Typical range: 20-80")

        with col2:
            st.markdown("#### Climate Conditions")
            temperature = st.slider(t('temperature'), 8.0, 44.0, 25.0, 0.5,
                                   help="Average temperature in °C")
            humidity = st.slider(t('humidity'), 14.0, 100.0, 71.0, 0.5,
                                help="Relative humidity in %")
            rainfall = st.slider(t('rainfall'), 20.0, 300.0, 103.0, 5.0,
                                help="Annual/seasonal rainfall in mm")

        with col3:
            st.markdown("#### Soil pH")
            ph = st.slider(t('ph'), 3.5, 9.5, 6.5, 0.1,
                          help="Soil pH level. Neutral = 7.0")

            ph_status = get_ph_status(ph)
            st.markdown(f"""
            <div style='margin-top:1rem; padding:0.9rem; background:#FFFFFF;
                        border-radius:8px; border-left:5px solid {ph_status['color']};
                        box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
                <b style='color:{ph_status['color']}; font-size:1rem'>pH {ph:.1f} - {ph_status['status']}</b><br>
                <span style='color:#333333; font-size:0.87rem'>{ph_status['advice']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Soil Summary")
            npk = get_npk_status(N, P, K)
            for nutrient, info in npk.items():
                color = info['color']
                st.markdown(f"""
                <span style='background:{color}; color:#FFFFFF; border:none;
                             padding:3px 10px; border-radius:10px; font-size:0.82rem;
                             margin:3px; display:inline-block; font-weight:600'>
                    {nutrient}: {info['status']}
                </span>
                """, unsafe_allow_html=True)

        submitted = st.form_submit_button(f"{t('predict_btn')}", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing soil and climate data..."):
            try:
                result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

                st.markdown("---")
                st.markdown(f"<h3 class='section-header'>Results</h3>", unsafe_allow_html=True)

                col_main, col_info = st.columns([1, 2])

                with col_main:
                    st.markdown(f"""
                    <div class='result-box' style='text-align:center'>
                        <div style='font-size:4rem'>{result['emoji']}</div>
                        <h2 style='color:White; margin:0.5rem 0; font-weight:700'>{result['crop'].upper()}</h2>
                        <div style='font-size:1.05rem; color:black; font-weight:600'>{t('confidence')}: {format_confidence(result['confidence'])}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if result['info']:
                        info = result['info']
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p><b>Season:</b> {info.get('season','N/A')}</p>
                            <p><b>Duration:</b> {info.get('duration','N/A')}</p>
                            <p><b>Water Need:</b> {info.get('water','N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)

                with col_info:
                    st.markdown(f"**{t('top_crops')}**")
                    for i, crop_data in enumerate(result['top3'], 1):
                        rank_emoji = ["1", "2", "3"][i-1]
                        pct = crop_data['probability'] * 100
                        st.markdown(f"""
                        <div class='crop-card' style='margin-bottom:0.8rem; text-align:left; padding:1rem'>
                            <span style='font-size:1.3rem'>{rank_emoji}</span>
                            <span style='font-size:1.5rem; margin:0 0.5rem'>{crop_data['emoji']}</span>
                            <b style='font-size:1.05rem; color:#0B3D1F'>{crop_data['crop'].upper()}</b>
                            <div style='margin-top:0.5rem'>
                                <div style='background:#D5E8D4; border-radius:10px; height:10px; overflow:hidden'>
                                    <div style='background:{"#145A32" if i==1 else "#27AE60"};
                                               height:100%; width:{min(pct,100):.0f}%; border-radius:10px'>
                                    </div>
                                </div>
                                <small style='color:#333333; font-weight:500'>{pct:.1f}% confidence</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Soil health summary
                st.markdown("---")
                st.markdown("**Soil Health Analysis**")
                soil = get_soil_health(N, P, K, ph)
                cols = st.columns(4)
                labels = {'nitrogen': 'Nitrogen', 'phosphorus': 'Phosphorus',
                         'potassium': 'Potassium', 'ph': 'pH'}
                colors = {'Low': '#FD7E14', 'High': '#DC3545', 'Optimal': '#28A745',
                         'Acidic': '#FD7E14', 'Alkaline': '#FD7E14',
                         'Neutral/Optimal': '#28A745', 'Excess': '#DC3545'}
                for i, (key, label) in enumerate(labels.items()):
                    status = soil[key]
                    color = colors.get(status, '#6C757D')
                    with cols[i]:
                        st.markdown(f"""
                        <div style='text-align:center; padding:1rem; background:#FFFFFF;
                                    border-radius:10px; border:2px solid {color};
                                    box-shadow:0 2px 8px rgba(0,0,0,0.08)'>
                            <div style='font-size:1.5rem'>
                                {"N" if key=="nitrogen" else "P" if key=="phosphorus" else "K" if key=="potassium" else "pH"}
                            </div>
                            <b style='color:#111111'>{label}</b><br>
                            <span style='color:#FFFFFF; background:{color}; font-weight:700;
                                         padding:2px 10px; border-radius:8px; font-size:0.88rem;
                                         display:inline-block; margin-top:4px'>{status}</span>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}. Ensure data/crop_data.csv exists.")

    # Disclaimer
    st.markdown(f"""
    <div class='disclaimer-box'>
        {t('disclaimer')}
    </div>
    """, unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Disease Detection":
    st.markdown(f"<h2 class='section-header'>{t('disease_title')}</h2>", unsafe_allow_html=True)
    st.markdown("*Simple Gemini-based disease recognition from uploaded image.*")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown("#### Upload Leaf Image")
        uploaded_file = st.file_uploader(
            t('upload_image'),
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload a clear image of the affected plant leaf"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
            analyze_btn = st.button("Analyze with Gemini", use_container_width=True)
        else:
            st.markdown("""
            <div class='info-box'>
                <b>Tips for best results:</b><br>
                • Use clear daylight image<br>
                • Keep leaf in focus<br>
                • Capture visible disease spots<br>
                • Avoid blurry photos
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with col_result:
        if uploaded_file and analyze_btn:
            gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not gemini_api_key.strip():
                st.error("GEMINI_API_KEY not found. Please add it in your .env file.")
            else:
                with st.spinner("Analyzing image with Gemini..."):
                    try:
                        result = analyze_disease_image_with_gemini(image, gemini_api_key.strip())

                        st.markdown("""
                        <div class='healthy-box'>
                            <h3 style='color:#145A32'>Gemini Diagnosis</h3>
                            <p style='font-size:0.9rem'>Generated from uploaded image</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(result['response'])
                        st.caption(f"Model: {result['model']}")

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.info("Please try again with a clearer image or valid Gemini API key.")
        elif not uploaded_file:
            st.markdown("""
            <div style='text-align:center; padding:3rem; color:#444444;
                        background:#F7F9F7; border-radius:12px; border:2px dashed #AAAAAA'>
                <div style='font-size:4rem'>🔬</div>
                <p style='font-weight:600; color:#333333'>Upload a plant leaf image to start analysis</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Supported Crops"):
        crops = [
            "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas", "Moth Beans",
            "Mung Bean", "Black Gram", "Lentil", "Pomegranate", "Banana", "Mango",
            "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya",
            "Coconut", "Cotton", "Jute", "Coffee"
        ]
        st.markdown(", ".join(crops))

    st.markdown(f"""
    <div class='disclaimer-box'>
        {t('disclaimer')}
    </div>
    """, unsafe_allow_html=True)

# PAGE 3: WEATHER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "Weather & Advisory":
    st.markdown(f"<h2 class='section-header'>{t('weather_title')}</h2>", unsafe_allow_html=True)

    col_input, col_weather = st.columns([1, 2])

    with col_input:
        city = st.text_input(t('enter_city'), value="Pune",
                            placeholder="e.g. Mumbai, Nagpur, Delhi...")
        api_key = st.text_input(
            "OpenWeatherMap API Key (optional)",
            value="",
            type="password",
            help="Get FREE API key at openweathermap.org. Leave blank for demo mode."
        )
        get_btn = st.button(f"{t('get_weather')}", use_container_width=True)

        st.markdown("""
        <div class='info-box' style='margin-top:1rem'>
            <b>Free API Key:</b><br>
            Visit <a href='https://openweathermap.org/api' target='_blank'>openweathermap.org</a>
            to get your FREE API key (free tier: 60 calls/min).<br><br>
            Without a key, the app uses demo/mock weather data.
        </div>
        """, unsafe_allow_html=True)

        # Seasonal crops
        st.markdown("---")
        st.markdown("#### This Month's Crops")
        month = datetime.datetime.now().month
        seasonal = get_seasonal_crops(month)
        for crop in seasonal:
            st.markdown(f"- {crop}")

    with col_weather:
        if get_btn or city:
            with st.spinner("Fetching weather..."):
                try:
                    weather = get_weather(city, api_key if api_key else None)

                    icon = get_weather_icon_emoji(weather['description'])

                    st.markdown(f"""
                    <div class='weather-card'>
                        <div style='font-size:3rem'>{icon}</div>
                        <h2 style='margin:0.5rem 0'>{weather['city']}, {weather['country']}</h2>
                        <div style='font-size:3rem; font-weight:700'>{weather['temperature']}°C</div>
                        <p style='font-size:1.1rem; color:#E3F2FD; opacity:1'>{weather['description']}</p>
                        <p style='color:#BBDEFB; opacity:1; font-size:0.85rem'>Feels like {weather['feels_like']}°C | Updated: {weather['timestamp']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Humidity", f"{weather['humidity']}%")
                    with m2:
                        st.metric("Wind", f"{weather['wind_speed']} km/h")
                    with m3:
                        st.metric("Visibility", f"{weather['visibility']} km")
                    with m4:
                        st.metric("Pressure", f"{weather['pressure']} hPa")

                    # Alerts
                    if weather['alerts']:
                        st.markdown("#### Weather Alerts")
                        for alert in weather['alerts']:
                            st.markdown(f"""
                            <div class='danger-box'>
                                {alert}
                            </div>
                            """, unsafe_allow_html=True)

                    # Farming advisory
                    st.markdown("#### Farming Advisory")
                    for advice in weather['farming_advice']:
                        st.markdown(f"""
                        <div style='background:#EAFAF1; padding:0.7rem 1rem; border-radius:8px;
                                    margin:0.4rem 0; border-left:4px solid #1E8449;
                                    font-size:0.92rem; color:#0B3D1F; font-weight:500'>
                            {advice}
                        </div>
                        """, unsafe_allow_html=True)

                    # Source note
                    st.markdown(f"""
                    <div style='background:#EEEEEE; padding:0.5rem 1rem; border-radius:8px;
                                font-size:0.82rem; color:#333333; margin-top:0.5rem; font-weight:500'>
                        Data source: {weather['source']}
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Weather fetch error: {e}")

    st.markdown(f"""
    <div class='disclaimer-box'>
        {t('disclaimer')}
    </div>
    """, unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 4: CHATBOT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "Farm Chatbot":
    st.markdown(f"<h2 class='section-header'>{t('chatbot_title')}</h2>", unsafe_allow_html=True)
    st.markdown(f"*{t('chatbot_desc')}*")

    # Quick action buttons
    st.markdown("**Quick Questions:**")
    quick_cols = st.columns(4)
    quick_questions = [
        "What crops for Kharif season?",
        "How to prevent leaf blight?",
        "Tell me about fertilizers",
        "Government farming schemes"
    ]
    quick_responses = []
    for i, q in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                quick_responses.append(q)

    for q in quick_responses:
        st.session_state.chat_history.append({"role": "user", "content": q})
        response = get_chatbot_response(q, st.session_state.language)
        st.session_state.chat_history.append({"role": "bot", "content": response})

    # Chat display area
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style='text-align:center; padding:2.5rem; color:#333333;
                        background:#F7F9F7; border-radius:12px; border:2px dashed #AAAAAA'>
                <div style='font-size:3rem'>AI</div>
                <p style='font-weight:700; color:#111111; font-size:1.05rem'>Welcome to the Farm Assistant Chatbot!</p>
                <p style='color:#444444'>Ask me about crops, diseases, fertilizers, irrigation, weather, or government schemes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-message-user'>
                        <b>You:</b> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-message-bot'>
                        <b>AgriBot:</b><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

    # Input area
    st.markdown("---")
    col_input, col_send, col_clear = st.columns([6, 1, 1])

    with col_input:
        user_input = st.text_input(
            "Message",
            placeholder=t('type_message'),
            label_visibility="collapsed",
            key="chat_input"
        )

    with col_send:
        send = st.button("Send", use_container_width=True, help="Send message")

    with col_clear:
        if st.button("Clear", use_container_width=True, help="Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if (send or user_input) and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = get_chatbot_response(user_input, st.session_state.language)
        st.session_state.chat_history.append({"role": "bot", "content": response})
        st.rerun()

    st.markdown(f"""
    <div class='disclaimer-box'>
        {t('disclaimer')}
    </div>
    """, unsafe_allow_html=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 5: ABOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "About":
    st.markdown("<h2 class='section-header'>About This Application</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background:#FFFFFF; border-left:5px solid #1E8449; border-radius:12px;
                    padding:1.2rem 1.5rem; margin-bottom:1.2rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.10)'>
            <h3 style='color:#145A32; font-size:1.3rem; margin:0 0 0.4rem'>ðŸŒ¾ AI Smart Agriculture Assistant</h3>
            <p style='color:#222222; font-size:0.95rem; margin:0'>
                A comprehensive, <b>100% FREE</b> AI-powered tool built for Indian farmers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#FFFFFF; border-radius:12px; padding:1.2rem 1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.10); margin-bottom:1.2rem'>
            <h4 style='color:#145A32; margin:0 0 0.8rem'>ðŸš€ Features</h4>
            <p style='color:#111111; margin:0.35rem 0'>ðŸŒ± <b style="color:#145A32">Crop Recommendation:</b> <span style="color:#222222">RandomForest ML â€” 22 crops</span></p>
            <p style='color:#111111; margin:0.35rem 0'>ðŸ”¬ <b style="color:#145A32">Disease Detection:</b> <span style="color:#222222">Gemini image analysis</span></p>
            <p style='color:#111111; margin:0.35rem 0'>ðŸ’Š <b style="color:#145A32">Medicine Database:</b> <span style="color:#222222">Curated pesticide & treatment DB</span></p>
            <p style='color:#111111; margin:0.35rem 0'>ðŸŒ¤ï¸ <b style="color:#145A32">Weather Module:</b> <span style="color:#222222">Real API + farming advisories</span></p>
            <p style='color:#111111; margin:0.35rem 0'>ðŸ¤– <b style="color:#145A32">Chatbot:</b> <span style="color:#222222">Rule-based agriculture Q&A bot</span></p>
            <p style='color:#111111; margin:0.35rem 0'>ðŸŒ <b style="color:#145A32">Multilingual:</b> <span style="color:#222222">English + Marathi support</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#FFFFFF; border-radius:12px; padding:1.2rem 1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.10); margin-bottom:1.2rem'>
            <h4 style='color:#145A32; margin:0 0 0.8rem'>ðŸ› ï¸ Technology Stack</h4>
            <table style='width:100%; border-collapse:collapse; font-size:0.92rem'>
                <tr style='background:#EAFAF1'>
                    <th style='padding:7px 10px; text-align:left; color:#0B3D1F; border:1px solid #A9DFBF'>Component</th>
                    <th style='padding:7px 10px; text-align:left; color:#0B3D1F; border:1px solid #A9DFBF'>Technology</th>
                </tr>
                <tr>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Frontend</td>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Streamlit</td>
                </tr>
                <tr style='background:#F7FBF7'>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Crop ML</td>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>RandomForest (scikit-learn)</td>
                </tr>
                <tr>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Disease AI</td>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Gemini API (Vision)</td>
                </tr>
                <tr style='background:#F7FBF7'>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Weather</td>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>OpenWeatherMap (Free Tier)</td>
                </tr>
                <tr>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Language</td>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Python 3.8+</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1E1E1E; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1.2rem'>
            <h4 style='color:#A9DFBF; margin:0 0 0.6rem'>ðŸ“ Project Structure</h4>
            <pre style='color:#E8F5E9; font-size:0.82rem; margin:0; line-height:1.7; font-family:monospace'>smart_agri/
â”œâ”€â”€ app.py              â† Main application
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ crop_model.py   â† Crop ML model
â”‚   â””â”€â”€ disease_model.pyâ† Disease CNN
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ weather.py      â† Weather module
â”‚   â”œâ”€â”€ chatbot.py      â† Chatbot engine
â”‚   â””â”€â”€ helpers.py      â† Utilities
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ crop_data.csv   â† Training dataset
â”‚   â”œâ”€â”€ medicine_db.jsonâ† Treatment database
â”‚   â””â”€â”€ translations.json
â””â”€â”€ requirements.txt</pre>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h4 style='color:#145A32; font-weight:700'>ðŸ“Š Model Information</h4>", unsafe_allow_html=True)

        try:
            from models.crop_model import load_model
            model, scaler, le, acc = load_model()
            accuracy_val = f"{acc:.1%}" if acc else "Loaded âœ…"
            st.markdown(f"""
            <div style='background:#EAFAF1; border:2px solid #1E8449; border-radius:12px;
                        padding:1rem 1.5rem; margin-bottom:1rem'>
                <p style='color:#555555; font-size:0.85rem; margin:0'>ðŸŽ¯ Crop Model Accuracy</p>
                <p style='color:#0B3D1F; font-size:2rem; font-weight:800; margin:0.2rem 0'>{accuracy_val}</p>
                <p style='color:#1A4D2E; font-size:0.85rem; margin:0'>Trained on 22 crop classes</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div style='background:#EBF5FB; border:2px solid #2E86C1; border-radius:12px; padding:1rem 1.5rem'>
                <p style='color:#1A3C54; margin:0'>â„¹ï¸ Run Crop Prediction to initialize the model</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#FFFFFF; border-radius:12px; padding:1.2rem 1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.10); margin-bottom:1.2rem'>
            <h4 style='color:#145A32; margin:0 0 0.8rem'>ðŸ’° Cost: Completely FREE</h4>
            <p style='color:#111111; margin:0.3rem 0'>âœ… No paid APIs required</p>
            <p style='color:#111111; margin:0.3rem 0'>âœ… No cloud services needed</p>
            <p style='color:#111111; margin:0.3rem 0'>âœ… Runs 100% on local system</p>
            <p style='color:#111111; margin:0.3rem 0'>âœ… Open-source libraries only</p>
            <p style='color:#111111; margin:0.3rem 0'>âœ… Free weather API tier</p>
            <p style='color:#111111; margin:0.3rem 0; font-size:1.1rem; font-weight:700; color:#145A32'>Total Cost: â‚¹0 / $0</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#FFFFFF; border-radius:12px; padding:1.2rem 1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.10); margin-bottom:1.2rem'>
            <h4 style='color:#145A32; margin:0 0 0.8rem'>ðŸ“ž Farmer Helplines</h4>
            <table style='width:100%; border-collapse:collapse; font-size:0.92rem'>
                <tr style='background:#EAFAF1'>
                    <th style='padding:7px 10px; text-align:left; color:#0B3D1F; border:1px solid #A9DFBF'>Service</th>
                    <th style='padding:7px 10px; text-align:left; color:#0B3D1F; border:1px solid #A9DFBF'>Number</th>
                </tr>
                <tr>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Kisan Call Center</td>
                    <td style='padding:6px 10px; color:#145A32; font-weight:700; border:1px solid #D5E8D4'>1800-180-1551</td>
                </tr>
                <tr style='background:#F7FBF7'>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>PM-KISAN Helpline</td>
                    <td style='padding:6px 10px; color:#145A32; font-weight:700; border:1px solid #D5E8D4'>155261</td>
                </tr>
                <tr>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>KVK Helpline</td>
                    <td style='padding:6px 10px; color:#145A32; font-weight:700; border:1px solid #D5E8D4'>1800-425-1122</td>
                </tr>
                <tr style='background:#F7FBF7'>
                    <td style='padding:6px 10px; color:#111111; border:1px solid #D5E8D4'>Crop Insurance</td>
                    <td style='padding:6px 10px; color:#145A32; font-weight:700; border:1px solid #D5E8D4'>1800-200-7710</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#FFFFFF; border-radius:12px; padding:1.2rem 1.5rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.10)'>
            <h4 style='color:#145A32; margin:0 0 0.8rem'>ðŸ”— Useful Resources</h4>
            <p style='margin:0.4rem 0'><a href='https://pmkisan.gov.in' target='_blank'
               style='color:#1565C0; font-weight:600; text-decoration:none'>ðŸ›ï¸ PM-KISAN Portal</a></p>
            <p style='margin:0.4rem 0'><a href='https://enam.gov.in' target='_blank'
               style='color:#1565C0; font-weight:600; text-decoration:none'>ðŸ›’ eNAM Online Market</a></p>
            <p style='margin:0.4rem 0'><a href='https://soilhealth.dac.gov.in' target='_blank'
               style='color:#1565C0; font-weight:600; text-decoration:none'>ðŸŒ Soil Health Card</a></p>
            <p style='margin:0.4rem 0'><a href='https://agmarknet.gov.in' target='_blank'
               style='color:#1565C0; font-weight:600; text-decoration:none'>ðŸ“ˆ AgMarkNet (Mandi Prices)</a></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='disclaimer-box'>
        {t('disclaimer')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div style='text-align:center; margin-top:20px; padding:10px;
            color:#555; font-size:2.0rem;'>
    ðŸ‘¨â€ðŸ’» Developed by <b style='color:#145A32;'>Rushikesh Gangawar</b>
</div>
""", unsafe_allow_html=True)
    

st.markdown("""
<style>

/* your existing CSS */

/* ðŸ”¥ Predict Button Color Fix */
div[data-testid="stFormSubmitButton"] button {
    background: #1976D2 !important;
    color: #FFFFFF !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #0D47A1 !important;
}

</style>
""", unsafe_allow_html=True)


