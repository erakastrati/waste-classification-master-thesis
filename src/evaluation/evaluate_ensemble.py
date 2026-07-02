"""
evaluate_ensemble.py  —  EXP-010: Ensemble + TTA evaluation

Compares single-model TTA vs 3-model ensemble TTA on real_test_dataset.

Run from project root:
    python -m src.evaluation.evaluate_ensemble
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import load_model

from src.inference.tta_predict import predict_with_tta
from src.inference.ensemble_predict import load_ensemble_models, predict_ensemble_tta

HOUSEHOLD_MODEL = "results/efficientnet_household/efficientnet_household.keras"
DATASET_PATH    = "data/real_test_dataset"
IMAGE_SIZE      = (224, 224)
CLASS_NAMES     = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
OUTPUT_DIR      = "results/ensemble_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset():
    images, labels = [], []
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
    return images, np.array(labels)


def save_results(name, y_true, y_pred):
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
    safe = name.lower().replace(" ", "_").replace("+", "")
    plt.savefig(f"{OUTPUT_DIR}/{safe}_confusion_matrix.png")
    plt.close()

    with open(f"{OUTPUT_DIR}/{safe}_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\n\n{report}")

    return acc, macro_f1


print("Loading models...")
household_model = load_model(HOUSEHOLD_MODEL)
ensemble_models = load_ensemble_models()

print("Loading real-world test set...")
images, y_true = load_dataset()
print(f"Images: {len(images)}")

y_pred_single = []
y_pred_ensemble = []

for i, img in enumerate(images):
    if (i + 1) % 5 == 0:
        print(f"  [{i + 1}/{len(images)}]")
    y_pred_single.append(int(np.argmax(predict_with_tta(household_model, img, IMAGE_SIZE))))
    y_pred_ensemble.append(int(np.argmax(predict_ensemble_tta(ensemble_models, img, IMAGE_SIZE))))

y_pred_single = np.array(y_pred_single)
y_pred_ensemble = np.array(y_pred_ensemble)

print("\n" + "=" * 50)
print("EXP-010: Ensemble + TTA vs Household + TTA (78 photos)")
print("=" * 50)

acc_single, f1_single = save_results("Household TTA", y_true, y_pred_single)
acc_ens, f1_ens = save_results("Ensemble TTA", y_true, y_pred_ensemble)

print(f"\nHousehold + TTA:  {acc_single*100:.2f}%  (macro F1 {f1_single:.2f})")
print(f"Ensemble + TTA:   {acc_ens*100:.2f}%  (macro F1 {f1_ens:.2f})")
print(f"Delta:            {(acc_ens - acc_single)*100:+.2f}pp")

fixed = int(np.sum((y_pred_single != y_true) & (y_pred_ensemble == y_true)))
broken = int(np.sum((y_pred_single == y_true) & (y_pred_ensemble != y_true)))
print(f"\nFixed:  {fixed}")
print(f"Broken: {broken}")

labels = ["Household + TTA", "Ensemble + TTA"]
accs = [acc_single * 100, acc_ens * 100]
plt.figure(figsize=(6, 4))
plt.bar(labels, accs, color=["#64748b", "#3b82f6"])
plt.ylim(0, 100)
plt.ylabel("Real-World Accuracy (%)")
plt.title("EXP-010: Ensemble Impact")
for i, v in enumerate(accs):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png")
plt.close()

print(f"\nArtifacts saved to {OUTPUT_DIR}/")
