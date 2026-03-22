"""
=====================================================
  Abusive Word Detection - Flask Backend
  Author: B.Tech NLP Mini Project
  Description: REST API server that loads the trained
               ML model and serves predictions via /predict
=====================================================
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re
import string
import os

# ─────────────────────────────────────────────────
#  App Initialization
# ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


# ─────────────────────────────────────────────────
#  Load Saved Model & Vectorizer
# ─────────────────────────────────────────────────
def load_model():
    """Load trained Logistic Regression model and TF-IDF vectorizer."""
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "Model files not found! Please run `python train_model.py` first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


try:
    model, vectorizer = load_model()
    print("✅ Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    print(f"⚠️  Warning: {e}")
    model, vectorizer = None, None


# ─────────────────────────────────────────────────
#  Text Preprocessing (same as training)
# ─────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Preprocess the input text before prediction:
    - Lowercase
    - Remove digits
    - Remove punctuation
    - Strip extra whitespace
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# ─────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Request Body (JSON): { "text": "your input here" }
    Response (JSON):
    {
        "label": "Abusive" | "Not Abusive",
        "confidence": <float 0-100>,
        "original_text": "<input>",
        "processed_text": "<cleaned_input>"
    }
    """
    # ── Check if model is loaded ──
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first."
        }), 500

    # ── Parse request ──
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400

    user_input = data["text"].strip()

    if not user_input:
        return jsonify({"error": "Input text cannot be empty."}), 400

    # ── Preprocess & Predict ──
    cleaned = preprocess_text(user_input)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Class 1 = Abusive, Class 0 = Not Abusive
    label = "Abusive" if prediction == 1 else "Not Abusive"
    confidence = round(float(probabilities[prediction]) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "original_text": user_input,
        "processed_text": cleaned,
        "abusive_prob": round(float(probabilities[1]) * 100, 2),
        "safe_prob": round(float(probabilities[0]) * 100, 2)
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# ─────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Abusive Word Detector API...")
    print("   URL: http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
