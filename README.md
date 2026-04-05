# 🌾 AI Smart Agriculture Assistant

A comprehensive, **100% FREE** AI-powered agriculture assistant for Indian farmers.
Built with Python, Streamlit, Machine Learning, and Deep Learning.

---

## 🚀 Features

| Feature | Description | Technology |
|---------|-------------|------------|
| 🌱 **Crop Recommendation** | Predict best crop from soil & climate data | RandomForest ML |
| 🔬 **Disease Detection** | Detect plant diseases from leaf photos | CNN / MobileNetV2 |
| 💊 **Medicine Database** | Pesticide recommendations with dosage | Local JSON DB |
| 🌤️ **Weather Module** | City weather + farming advisory | OpenWeatherMap (Free) |
| 🤖 **Farm Chatbot** | Rule-based agricultural Q&A bot | Pattern Matching |
| 🌐 **Multilingual** | English + Marathi support | Translation JSON |

---

## 📁 Project Structure

```
smart_agri/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── models/
│   ├── __init__.py
│   ├── crop_model.py         ← RandomForest crop prediction
│   └── disease_model.py      ← CNN plant disease detection
│
├── utils/
│   ├── __init__.py
│   ├── weather.py            ← Weather API + mock system
│   ├── chatbot.py            ← Rule-based chatbot
│   └── helpers.py            ← Utility functions
│
└── data/
    ├── crop_data.csv         ← Training dataset (22 crops)
    ├── medicine_db.json      ← Disease treatment database
    └── translations.json     ← English + Marathi translations
```

---

## 🛠️ Installation & Setup

### Step 1: Prerequisites
Ensure you have **Python 3.8+** installed:
```bash
python --version
```

### Step 2: Download / Clone the Project
```bash
# If using git:
git clone <your-repo-url>
cd smart_agri

# Or extract the zip file and navigate to the folder
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## 🌐 Optional: Enable Real Weather (Free API Key)

1. Visit **https://openweathermap.org/api**
2. Click **"Sign Up"** (completely FREE)
3. Go to **API Keys** tab in your account
4. Copy your API key
5. In the app → **Weather tab** → paste your API key

> Without the API key, the app uses mock/demo weather data.

---

## 🧠 Optional: Full AI Disease Detection

For actual CNN-based disease detection (instead of color analysis), install TensorFlow:

```bash
# Option 1: Full TensorFlow (requires ~500MB)
pip install tensorflow

# Option 2: CPU-only version (lighter)
pip install tensorflow-cpu
```

### Training the Disease Model (Advanced)
If you have the PlantVillage dataset from Kaggle:
```bash
# Download from: https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to: data/PlantVillage/

python train_disease_model.py  # (advanced - requires GPU for faster training)
```

---

## 📊 Machine Learning Details

### Crop Recommendation Model
- **Algorithm:** Random Forest Classifier (200 trees)
- **Features:** N, P, K, Temperature, Humidity, pH, Rainfall
- **Classes:** 22 crops (rice, wheat, maize, cotton, etc.)
- **Expected Accuracy:** ~98-99% on validation set
- **Dataset:** Synthetic + augmented crop dataset

### Disease Detection
- **Mode 1 (Default):** Color analysis heuristics (no install needed)
- **Mode 2 (Full):** MobileNetV2 CNN (requires TensorFlow)
- **Classes:** 38 disease categories (PlantVillage standard)
- **Input:** 224×224 RGB leaf images

---

## 💰 Cost Breakdown

| Component | Cost |
|-----------|------|
| Python + Streamlit | ✅ FREE |
| scikit-learn ML | ✅ FREE |
| TensorFlow (optional) | ✅ FREE |
| OpenWeatherMap API | ✅ FREE (60 calls/min) |
| Dataset | ✅ FREE |
| Hosting (local) | ✅ FREE |
| **TOTAL** | **₹0 / $0** |

---

## 🌾 Supported Crops (Crop Recommendation)
Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans,
Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango,
Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya,
Coconut, Cotton, Jute, Coffee

## 🔬 Supported Diseases (Disease Detection)
Apple (Scab, Black Rot, Cedar Rust), Corn/Maize (Gray Leaf Spot, Common Rust, Northern Blight),
Grapes (Black Rot, Esca, Leaf Blight), Potato (Early Blight, Late Blight),
Tomato (8 diseases), and more — **38 total classes**

---

## 📞 Farmer Helplines

| Service | Number |
|---------|--------|
| Kisan Call Center | 1800-180-1551 |
| PM-KISAN Helpline | 155261 |
| KVK Network | 1800-425-1122 |
| Crop Insurance | 1800-200-7710 |

---

## ⚠️ Disclaimer

> **Always consult a qualified agriculture expert or local Krishi Vigyan Kendra (KVK) before using any pesticide or making major farming decisions. This tool is for educational and advisory purposes only.**

---

## 🤝 Contributing

This project is designed for the benefit of Indian farmers. Contributions welcome!
- Add more diseases to the database
- Improve the chatbot knowledge base
- Add more language translations
- Enhance the ML model accuracy

---

*Built with ❤️ for Indian Farmers | 100% Free & Open Source*
