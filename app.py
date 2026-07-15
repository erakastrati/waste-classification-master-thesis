import os
import io
import base64
import numpy as np

os.environ.setdefault("FLASK_SKIP_DOTENV", "1")

from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.models import load_model
from src.inference.tta_predict import predict_with_tta

app = Flask(__name__)

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "results/efficientnet_garbage/efficientnet_garbage_best.keras"
IMAGE_SIZE  = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_INFO = {
    "cardboard": {
        "icon": "📦",
        "color": "#8B6914",
        "tip": "Palos kartonin dhe hiq shiritin ngjites para riciklimit"
    },
    "glass": {
        "icon": "🍶",
        "color": "#4A90D9",
        "tip": "Shpelaj enet prej qelqi para se te hedhesh ne koshin per qelq"
    },
    "metal": {
        "icon": "🥫",
        "color": "#7F8C8D",
        "tip": "Zbraz dhe shperlaj kanaqet. Hiq etiketat nese eshte e mundur"
    },
    "paper": {
        "icon": "📄",
        "color": "#27AE60",
        "tip": "Mbaje letren te thate. Letra e yndyrshme (si kutite e picave) hidhet te mbeturinat e pergjithshm"
    },
    "plastic": {
        "icon": "♻️",
        "color": "#E74C3C",
        "tip": "Kontrollo numrin e riciklimit ne fund te produktit para ndarjes"
    },
    "trash": {
        "icon": "🗑️",
        "color": "#95A5A6",
        "tip": "Ky objekt nuk mund te riciklohet. Hidhe te mbeturinat e pergjithshme"
    },
}

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")


# ==========================================
# Routes
# ==========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img = Image.open(file.stream).convert("RGB")

        predictions = predict_with_tta(model, img, IMAGE_SIZE)
        confidence_scores = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
        confidence      = round(float(np.max(predictions)) * 100, 2)
        info            = CLASS_INFO[predicted_class]

        # Encode thumbnail for display
        buf = io.BytesIO()
        img.resize((300, 300)).save(buf, format="JPEG", quality=85)
        encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "all_scores":      confidence_scores,
            "icon":            info["icon"],
            "color":           info["color"],
            "tip":             info["tip"],
            "image_b64":       encoded_img,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
