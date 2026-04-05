"""
Crop Recommendation Model
Trains a Random Forest Classifier on crop data
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'crop_scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'crop_encoder.pkl')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'crop_data.csv')

CROP_EMOJIS = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫘', 'mothbeans': '🫘', 'mungbean': '🫘', 'blackgram': '🫘',
    'lentil': '🫘', 'pomegranate': '🍎', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍏',
    'orange': '🍊', 'papaya': '🧡', 'coconut': '🥥', 'cotton': '🌿',
    'jute': '🌿', 'coffee': '☕'
}

CROP_INFO = {
    'rice': {'season': 'Kharif', 'duration': '120-150 days', 'water': 'High'},
    'maize': {'season': 'Kharif/Rabi', 'duration': '90-120 days', 'water': 'Medium'},
    'chickpea': {'season': 'Rabi', 'duration': '90-100 days', 'water': 'Low'},
    'kidneybeans': {'season': 'Kharif', 'duration': '90-120 days', 'water': 'Medium'},
    'pigeonpeas': {'season': 'Kharif', 'duration': '150-180 days', 'water': 'Low'},
    'mothbeans': {'season': 'Kharif', 'duration': '75-90 days', 'water': 'Very Low'},
    'mungbean': {'season': 'Kharif', 'duration': '60-75 days', 'water': 'Low'},
    'blackgram': {'season': 'Kharif', 'duration': '75-90 days', 'water': 'Low'},
    'lentil': {'season': 'Rabi', 'duration': '100-120 days', 'water': 'Low'},
    'pomegranate': {'season': 'Perennial', 'duration': '5-7 months (fruit)', 'water': 'Low'},
    'banana': {'season': 'Perennial', 'duration': '9-12 months', 'water': 'High'},
    'mango': {'season': 'Perennial', 'duration': '3-5 months (fruit)', 'water': 'Medium'},
    'grapes': {'season': 'Perennial', 'duration': '6-8 months (fruit)', 'water': 'Medium'},
    'watermelon': {'season': 'Summer', 'duration': '70-90 days', 'water': 'High'},
    'muskmelon': {'season': 'Summer', 'duration': '70-80 days', 'water': 'Medium'},
    'apple': {'season': 'Perennial', 'duration': '5-7 months (fruit)', 'water': 'Medium'},
    'orange': {'season': 'Perennial', 'duration': '7-8 months (fruit)', 'water': 'Medium'},
    'papaya': {'season': 'Perennial', 'duration': '9-10 months', 'water': 'High'},
    'coconut': {'season': 'Perennial', 'duration': '6-12 months (fruit)', 'water': 'High'},
    'cotton': {'season': 'Kharif', 'duration': '150-180 days', 'water': 'Medium'},
    'jute': {'season': 'Kharif', 'duration': '100-120 days', 'water': 'High'},
    'coffee': {'season': 'Perennial', 'duration': '7-9 months (fruit)', 'water': 'High'},
}


def train_model():
    """Train the crop recommendation model"""
    df = pd.read_csv(DATA_PATH)

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
    y = df['label'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model artifacts
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

    return accuracy, model, scaler, le


def load_model():
    """Load trained model or train if not exists"""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        accuracy, model, scaler, le = train_model()
        return model, scaler, le, accuracy

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    return model, scaler, le, None


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predict best crop with probabilities"""
    model, scaler, le, accuracy = load_model()

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    crop_name = le.inverse_transform([prediction])[0]

    # Top 3 predictions
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3 = [
        {
            'crop': le.inverse_transform([idx])[0],
            'probability': float(probabilities[idx]),
            'emoji': CROP_EMOJIS.get(le.inverse_transform([idx])[0], '🌱'),
            'info': CROP_INFO.get(le.inverse_transform([idx])[0], {})
        }
        for idx in top3_idx
    ]

    return {
        'crop': crop_name,
        'confidence': float(probabilities[prediction]),
        'emoji': CROP_EMOJIS.get(crop_name, '🌱'),
        'info': CROP_INFO.get(crop_name, {}),
        'top3': top3
    }


def get_soil_health(N, P, K, ph):
    """Return soil health indicators"""
    health = {}
    health['nitrogen'] = 'Low' if N < 40 else ('High' if N > 80 else 'Optimal')
    health['phosphorus'] = 'Low' if P < 30 else ('High' if P > 70 else 'Optimal')
    health['potassium'] = 'Low' if K < 30 else ('High' if K > 80 else 'Optimal')
    health['ph'] = 'Acidic' if ph < 5.5 else ('Alkaline' if ph > 7.5 else 'Neutral/Optimal')
    return health


if __name__ == '__main__':
    accuracy, *_ = train_model()
    print(f"Model trained! Accuracy: {accuracy:.4f}")
