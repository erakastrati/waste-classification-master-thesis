"""
evaluate_tta.py  —  EXP-009: Test-Time Augmentation evaluation

Compares single-pass vs TTA inference on the real_test_dataset (78 photos).

Run from project root:
    python -m src.evaluation.evaluate_tta
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.inference.tta_predict import predict_with_tta

MODEL_PATH   = "results/efficientnet_household/efficientnet_household.keras"
DATASET_PATH = "data/real_test_dataset"
IMAGE_SIZE   = (224, 224)
CLASS_NAMES  = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
OUTPUT_DIR   = "results/tta_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset():
    images, labels, paths = [], [], []
    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATASET_PATH, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(cls_dir, fname)
            images.append(Image.open(path).convert("RGB"))
            labels.append(label_idx)
            paths.append(path)
    return images, np.array(labels), paths


def predict_single(model, img):
    arr = np.expand_dims(
        preprocess_input(np.array(img.resize(IMAGE_SIZE), dtype=np.float32)),
        axis=0,
    )
    return model.predict(arr, verbose=0)[0]


def evaluate(name, y_true, y_pred, model_obj):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Real-World Confusion Matrix — {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/{safe}_confusion_matrix.png")
    plt.close()

    with open(f"{OUTPUT_DIR}/{safe}_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}")

    return acc, macro_f1


print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading real-world test set...")
images, y_true, paths = load_dataset()
print(f"Images: {len(images)}")

y_pred_single = []
y_pred_tta = []

for i, img in enumerate(images):
    if (i + 1) % 10 == 0:
        print(f"  [{i + 1}/{len(images)}]")
    y_pred_single.append(int(np.argmax(predict_single(model, img))))
    y_pred_tta.append(int(np.argmax(predict_with_tta(model, img, IMAGE_SIZE))))

y_pred_single = np.array(y_pred_single)
y_pred_tta = np.array(y_pred_tta)

print("\n" + "=" * 50)
print("EXP-009: TTA vs Single-Pass (Household model, 78 photos)")
print("=" * 50)

acc_single, f1_single = evaluate("Single Pass", y_true, y_pred_single, model)
acc_tta, f1_tta = evaluate("TTA", y_true, y_pred_tta, model)

print(f"\nSingle pass: {acc_single*100:.2f}%  (macro F1 {f1_single:.2f})")
print(f"TTA:         {acc_tta*100:.2f}%  (macro F1 {f1_tta:.2f})")
print(f"Delta:       {(acc_tta - acc_single)*100:+.2f}pp")

changed = int(np.sum(y_pred_single != y_pred_tta))
fixed = int(np.sum((y_pred_single != y_true) & (y_pred_tta == y_true)))
broken = int(np.sum((y_pred_single == y_true) & (y_pred_tta != y_true)))
print(f"\nPredictions changed by TTA: {changed}")
print(f"  Fixed (was wrong → now correct): {fixed}")
print(f"  Broken (was correct → now wrong): {broken}")

# Bar chart comparison
labels = ["Single Pass", "TTA"]
accs = [acc_single * 100, acc_tta * 100]
plt.figure(figsize=(6, 4))
plt.bar(labels, accs, color=["#64748b", "#22c55e"])
plt.ylim(0, 100)
plt.ylabel("Real-World Accuracy (%)")
plt.title("EXP-009: TTA Impact on Real-World Test")
for i, v in enumerate(accs):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png")
plt.close()

print(f"\nArtifacts saved to {OUTPUT_DIR}/")
