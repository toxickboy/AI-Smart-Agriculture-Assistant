"""
Gemini-powered plant disease analysis.
Uses the official Google Gen AI SDK (google-genai) — the latest best practice.

Frontend image upload  ->  one Gemini API call  ->  text response.
"""

import io
import os
from typing import Dict

from PIL import Image

# ── Official Google Gen AI SDK ─────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Supported crops ────────────────────────────────────────────────────────
SUPPORTED_CROPS = [
    "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas", "Moth Beans",
    "Mung Bean", "Black Gram", "Lentil", "Pomegranate", "Banana", "Mango",
    "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya",
    "Coconut", "Cotton", "Jute", "Coffee",
]

# Ordered fallback list – first available model wins.
_MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def _pil_to_bytes_jpeg(image: Image.Image) -> bytes:
    """Convert a PIL Image to JPEG bytes (RGB, quality 90)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_client(api_key: str) -> genai.Client:
    """Instantiate the Gen AI client with the caller-supplied API key."""
    return genai.Client(api_key=api_key)


def analyze_disease_image_with_gemini(
    image: Image.Image,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> Dict[str, str]:
    """
    Analyze a plant-leaf image for disease using the Gemini vision API.

    Parameters
    ----------
    image : PIL.Image.Image   – The leaf / plant image to analyse.
    api_key : str             – Gemini API key (from .env / environment).
    model : str               – Preferred model name (falls back automatically).

    Returns
    -------
    {"response": <markdown text>, "model": <model name used>}
    """
    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = _build_client(api_key)
    image_bytes = _pil_to_bytes_jpeg(image)
    crops_text = ", ".join(SUPPORTED_CROPS)

    prompt = (
        "You are an expert agricultural assistant. Carefully analyze this plant image and provide:\n"
        "1) **Crop Name** – must be from this list if identifiable: "
        f"{crops_text}.\n"
        "2) **Disease Name** – or 'Healthy' if no disease is visible.\n"
        "3) **Confidence** – Low / Medium / High.\n"
        "4) **Farmer Actions** – 3 concise, practical steps the farmer should take.\n\n"
        "If the image is unclear or not a plant, say so. Keep the response structured and practical."
    )

    # Build a candidate list, ensuring the preferred model is tried first.
    candidates = [model] + [m for m in _MODEL_CANDIDATES if m != model]

    last_error: Exception | None = None
    for model_name in candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt),
                ],
            )
            text = (response.text or "").strip()
            if not text:
                text = "No response text received from Gemini."
            return {"response": text, "model": model_name}

        except Exception as exc:  # noqa: BLE001
            # 404 → model not available for this key; try next.
            err_str = str(exc).lower()
            if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                last_error = exc
                continue
            # Any other error is surfaced immediately.
            raise

    raise RuntimeError(
        "No available Gemini model was found for your API key/project. "
        f"Tried: {', '.join(candidates)}. "
        f"Last error: {last_error}"
    )
