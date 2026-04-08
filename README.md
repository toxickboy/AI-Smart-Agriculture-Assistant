# AI Smart Agriculture Assistant

A Streamlit app for farmers with crop recommendation, weather advisory, chatbot, and Gemini-based disease recognition from leaf images.

## Features

- Crop recommendation (RandomForest model)
- Disease detection using Gemini Vision API (image upload -> API response)
- Weather and farming advisory (OpenWeatherMap)
- Farm chatbot
- English/Marathi support

## Project Structure

```text
AI-Smart-Agriculture-Assistant/
|-- app.py
|-- requirements.txt
|-- models/
|   |-- crop_model.py
|   |-- disease_model.py
|-- utils/
|   |-- gemini_disease.py
|   |-- weather.py
|   |-- chatbot.py
|   |-- helpers.py
|-- data/
|   |-- crop_data.csv
|   |-- medicine_db.json
|   |-- translations.json
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Gemini Disease Detection Setup

1. Create/get a Gemini API key from Google AI Studio.
2. Create a `.env` file in project root:

```bash
GEMINI_API_KEY="your_api_key_here"
```

3. Open **Disease Detection** page in app.
4. Upload leaf image and click **Analyze with Gemini**.
5. Gemini response is shown directly in the frontend.

## Supported Crops for Gemini Analysis

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.

## Disclaimer

This tool is for advisory use only. Please consult local agriculture experts or KVK before major treatment decisions.
