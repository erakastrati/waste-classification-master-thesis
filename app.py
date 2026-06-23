import os
import io
import base64
import numpy as np

os.environ.setdefault("FLASK_SKIP_DOTENV", "1")

from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# ==========================================
# Configuration
# ==========================================

MODEL_PATH  = "results/efficientnet_augmented/efficientnet_augmented.keras"
IMAGE_SIZE  = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_INFO = {
    "cardboard": {"icon": "📦", "color": "#8B6914", "tip": "Flatten boxes and remove tape before recycling."},
    "glass":     {"icon": "🍶", "color": "#4A90D9", "tip": "Rinse glass containers before placing in the glass bin."},
    "metal":     {"icon": "🥫", "color": "#7F8C8D", "tip": "Empty and rinse cans. Remove labels if possible."},
    "paper":     {"icon": "📄", "color": "#27AE60", "tip": "Keep paper dry. Greasy paper (pizza boxes) goes to general waste."},
    "plastic":   {"icon": "♻️",  "color": "#E74C3C", "tip": "Check the recycling number on the bottom before sorting."},
    "trash":     {"icon": "🗑️", "color": "#95A5A6", "tip": "This item cannot be recycled. Dispose in general waste."},
}

# Load model once at startup
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
        img_resized = img.resize(IMAGE_SIZE)

        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array, verbose=0)[0]
        confidence_scores = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        predicted_class = CLASS_NAMES[int(np.argmax(predictions))]
        confidence      = round(float(np.max(predictions)) * 100, 2)
        info            = CLASS_INFO[predicted_class]

        # Encode thumbnail for display
        buf = io.BytesIO()
        img_resized_display = img.resize((300, 300))
        img_resized_display.save(buf, format="JPEG", quality=85)
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
