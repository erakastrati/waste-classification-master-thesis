"""
evaluate_rembg.py  —  EXP-006: Background Removal Preprocessing

Evaluates EfficientNetB0 + TACO model on the real_test_dataset
with and without background removal (rembg), to quantify the improvement.

Run from project root:
    python -m src.evaluation.evaluate_rembg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from rembg import remove as remove_bg
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================================
# Config
# ==========================================

MODEL_PATH   = "results/efficientnet_taco/efficientnet_taco.keras"
DATASET_PATH = "data/real_test_dataset"
IMAGE_SIZE   = (224, 224)
CLASS_NAMES  = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
OUTPUT_DIR   = "results/rembg_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

def load_images(dataset_path, class_names, image_size, use_rembg=False):
    X, y, paths = [], [], []
    label = "with background removal" if use_rembg else "without background removal"
    print(f"\nLoading images ({label})...")

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_path, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                if use_rembg:
                    img_nobg = remove_bg(img)
                    white_bg = Image.new("RGB", img_nobg.size, (255, 255, 255))
                    white_bg.paste(img_nobg, mask=img_nobg.split()[3])
                    img = white_bg
                img = img.resize(image_size)
                X.append(np.array(img, dtype=np.float32))
                y.append(cls_idx)
                paths.append(fpath)
            except Exception as e:
                print(f"  Skipped {fpath}: {e}")
    return np.array(X), np.array(y, dtype=int), paths


# ==========================================
# Load Model
# ==========================================

print(f"Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ==========================================
# Evaluate WITHOUT background removal
# ==========================================

X_raw, y_true, _ = load_images(DATASET_PATH, CLASS_NAMES, IMAGE_SIZE, use_rembg=False)
X_no_bg_removal   = preprocess_input(X_raw.copy())
preds_no_rembg     = np.argmax(model.predict(X_no_bg_removal, batch_size=16, verbose=1), axis=1)
acc_no_rembg       = accuracy_score(y_true, preds_no_rembg)

print(f"\nAccuracy WITHOUT background removal: {acc_no_rembg:.4f} ({acc_no_rembg*100:.2f}%)")
print("\nClassification Report (no rembg):")
print(classification_report(y_true, preds_no_rembg, target_names=CLASS_NAMES))

# ==========================================
# Evaluate WITH background removal
# ==========================================

X_rembg, y_true2, _ = load_images(DATASET_PATH, CLASS_NAMES, IMAGE_SIZE, use_rembg=True)
X_rembg_proc         = preprocess_input(X_rembg.copy())
preds_rembg           = np.argmax(model.predict(X_rembg_proc, batch_size=16, verbose=1), axis=1)
acc_rembg             = accuracy_score(y_true2, preds_rembg)

print(f"\nAccuracy WITH background removal: {acc_rembg:.4f} ({acc_rembg*100:.2f}%)")
print("\nClassification Report (with rembg):")
print(classification_report(y_true2, preds_rembg, target_names=CLASS_NAMES))

# ==========================================
# Summary
# ==========================================

delta = acc_rembg - acc_no_rembg
print("\n" + "=" * 50)
print("BACKGROUND REMOVAL IMPACT SUMMARY")
print("=" * 50)
print(f"Without rembg:  {acc_no_rembg*100:.2f}%")
print(f"With rembg:     {acc_rembg*100:.2f}%")
print(f"Delta:          {delta*100:+.2f}pp")

# ==========================================
# Comparison Bar Chart
# ==========================================

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Without\nBackground Removal", "With\nBackground Removal (rembg)"],
    [acc_no_rembg * 100, acc_rembg * 100],
    color=["#475569", "#3b82f6"],
    width=0.5,
    edgecolor="white"
)
for bar, val in zip(bars, [acc_no_rembg * 100, acc_rembg * 100]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

ax.set_title("Real-World Accuracy: Background Removal Impact\n(EfficientNetB0 + TACO)", fontsize=13)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rembg_comparison.png", dpi=150)
plt.close()
print(f"\nChart saved: {OUTPUT_DIR}/rembg_comparison.png")

# ==========================================
# Confusion matrices side by side
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, preds, title in zip(
    axes,
    [preds_no_rembg, preds_rembg],
    ["Without Background Removal", "With Background Removal (rembg)"]
):
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()
print(f"Confusion matrices saved: {OUTPUT_DIR}/confusion_matrices.png")
