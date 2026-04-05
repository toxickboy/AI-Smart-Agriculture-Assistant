"""
Plant Disease Detection Model
Uses a CNN (MobileNetV2 architecture) for leaf disease classification.
Falls back to a rule-based system if no trained model is available.
"""

import os
import json
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'disease_model.h5')
MEDICINE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'medicine_db.json')

# PlantVillage-style class names (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

DISPLAY_NAMES = {
    'Apple___Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___Cedar_apple_rust': 'Cedar Apple Rust',
    'Apple___healthy': 'Healthy Apple',
    'Blueberry___healthy': 'Healthy Blueberry',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
    'Cherry_(including_sour)___healthy': 'Healthy Cherry',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Maize Gray Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Maize Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Maize Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Healthy Maize',
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight',
    'Grape___healthy': 'Healthy Grape',
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus Greening (HLB)',
    'Peach___Bacterial_spot': 'Peach Bacterial Spot',
    'Peach___healthy': 'Healthy Peach',
    'Pepper,_bell___Bacterial_spot': 'Pepper Bacterial Spot',
    'Pepper,_bell___healthy': 'Healthy Pepper',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Healthy Potato',
    'Raspberry___healthy': 'Healthy Raspberry',
    'Soybean___healthy': 'Healthy Soybean',
    'Squash___Powdery_mildew': 'Squash Powdery Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
    'Strawberry___healthy': 'Healthy Strawberry',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Healthy Tomato',
}


def load_medicine_db():
    with open(MEDICINE_DB_PATH, 'r') as f:
        return json.load(f)


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def build_cnn_model():
    """Build MobileNetV2-based CNN model"""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(len(CLASS_NAMES), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except ImportError:
        return None


def heuristic_disease_detection(image: Image.Image):
    """
    Heuristic-based disease detection using color analysis.
    Used when no trained model is available.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized).astype('float32')

    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])

    # Brown/rust detection
    brown_pixels = np.sum((img_array[:, :, 0] > 120) & (img_array[:, :, 1] < 100) & (img_array[:, :, 2] < 80))
    total_pixels = 224 * 224
    brown_ratio = brown_pixels / total_pixels

    # Yellow detection
    yellow_pixels = np.sum((img_array[:, :, 0] > 150) & (img_array[:, :, 1] > 130) & (img_array[:, :, 2] < 80))
    yellow_ratio = yellow_pixels / total_pixels

    # Dark spots detection
    dark_pixels = np.sum((img_array[:, :, 0] < 60) & (img_array[:, :, 1] < 60) & (img_array[:, :, 2] < 60))
    dark_ratio = dark_pixels / total_pixels

    # Green health
    green_pixels = np.sum((img_array[:, :, 1] > img_array[:, :, 0] + 20) & (img_array[:, :, 1] > img_array[:, :, 2] + 20))
    green_ratio = green_pixels / total_pixels

    results = []

    if green_ratio > 0.45:
        results.append({'class': 'Tomato___healthy', 'confidence': 0.55 + green_ratio * 0.3})
    if brown_ratio > 0.15:
        results.append({'class': 'Tomato___Early_blight', 'confidence': 0.45 + brown_ratio})
        results.append({'class': 'Potato___Early_blight', 'confidence': 0.40 + brown_ratio * 0.8})
    if yellow_ratio > 0.1:
        results.append({'class': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'confidence': 0.40 + yellow_ratio})
        results.append({'class': 'Corn_(maize)___Common_rust_', 'confidence': 0.35 + yellow_ratio * 0.8})
    if dark_ratio > 0.1:
        results.append({'class': 'Apple___Black_rot', 'confidence': 0.35 + dark_ratio})
        results.append({'class': 'Grape___Black_rot', 'confidence': 0.32 + dark_ratio * 0.7})
    if brown_ratio > 0.08 and dark_ratio > 0.05:
        results.append({'class': 'Potato___Late_blight', 'confidence': 0.38 + brown_ratio * 0.5})

    if not results:
        results = [
            {'class': 'Tomato___healthy', 'confidence': 0.40},
            {'class': 'Tomato___Early_blight', 'confidence': 0.25},
            {'class': 'Potato___Early_blight', 'confidence': 0.20},
        ]

    results.sort(key=lambda x: x['confidence'], reverse=True)
    results = results[:3]

    # Normalize confidences
    total = sum(r['confidence'] for r in results)
    for r in results:
        r['confidence'] = r['confidence'] / total

    return results


def detect_disease(image: Image.Image):
    """Main disease detection function"""
    medicine_db = load_medicine_db()

    # Try TF model first
    model = None
    using_model = False

    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            using_model = True
        except Exception:
            pass

    if using_model and model is not None:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array, verbose=0)[0]
        top3_idx = np.argsort(predictions)[::-1][:3]
        results = [
            {'class': CLASS_NAMES[i], 'confidence': float(predictions[i])}
            for i in top3_idx
        ]
    else:
        results = heuristic_disease_detection(image)

    # Enrich with display names and medicine info
    top_result = results[0]
    class_name = top_result['class']
    display_name = DISPLAY_NAMES.get(class_name, class_name.replace('___', ' - ').replace('_', ' '))
    is_healthy = 'healthy' in class_name.lower()

    # Find medicine - try exact match first, then partial
    medicine_info = medicine_db.get(class_name)
    if not medicine_info:
        for key in medicine_db:
            if key.lower() in class_name.lower() or class_name.lower() in key.lower():
                medicine_info = medicine_db[key]
                break
    if not medicine_info:
        medicine_info = medicine_db.get('healthy')

    return {
        'class': class_name,
        'display_name': display_name,
        'confidence': top_result['confidence'],
        'is_healthy': is_healthy,
        'medicine_info': medicine_info,
        'top3': [
            {
                'class': r['class'],
                'display_name': DISPLAY_NAMES.get(r['class'], r['class'].replace('___', ' - ').replace('_', ' ')),
                'confidence': r['confidence']
            }
            for r in results
        ],
        'method': 'AI Model' if using_model else 'Color Analysis (Demo Mode)'
    }


if __name__ == '__main__':
    # Quick test
    test_img = Image.new('RGB', (224, 224), color=(80, 130, 60))
    result = detect_disease(test_img)
    print(f"Detected: {result['display_name']} ({result['confidence']:.2%})")
