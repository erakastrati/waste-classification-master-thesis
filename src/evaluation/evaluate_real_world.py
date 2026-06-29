import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


os.makedirs("results/real_world_test", exist_ok=True)


# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "data/real_test_dataset"
IMAGE_SIZE   = (224, 224)
CLASS_NAMES  = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

MODELS = {
    "CNN Baseline":              {
        "path":       "results/cnn_baseline/cnn_baseline.keras",
        "preprocess": lambda x: x / 255.0,
    },
    "MobileNetV2":               {
        "path":       "results/mobilenet/mobilenet.keras",
        "preprocess": mobilenet_preprocess,
    },
    "EfficientNetB0":            {
        "path":       "results/efficientnet/efficientnet.keras",
        "preprocess": efficientnet_preprocess,
    },
    "EfficientNetB0 + Augment": {
        "path":       "results/efficientnet_augmented/efficientnet_augmented.keras",
        "preprocess": efficientnet_preprocess,
    },
    "EfficientNetB0 + TACO":    {
        "path":       "results/efficientnet_taco/efficientnet_taco.keras",
        "preprocess": efficientnet_preprocess,
    },
    "EfficientNetB0 + RealWaste": {
        "path":       "results/efficientnet_realwaste/efficientnet_realwaste.keras",
        "preprocess": efficientnet_preprocess,
    },
}


# ==========================================
# Load all images and true labels
# ==========================================

def load_real_dataset(dataset_path, class_names, image_size):
    images = []
    labels = []
    paths  = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert("RGB").resize(image_size)
            images.append(np.array(img, dtype=np.float32))
            labels.append(label_idx)
            paths.append(fpath)

    return np.array(images), np.array(labels), paths


X_raw, y_true, file_paths = load_real_dataset(DATASET_PATH, CLASS_NAMES, IMAGE_SIZE)

print(f"Loaded {len(X_raw)} real-world images across {len(CLASS_NAMES)} classes.\n")


# ==========================================
# Evaluate each model
# ==========================================

results_summary = {}

for model_name, cfg in MODELS.items():

    print(f"{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    model = load_model(cfg["path"])

    X = cfg["preprocess"](X_raw.copy())

    predictions = model.predict(X, batch_size=16, verbose=1)
    y_pred      = np.argmax(predictions, axis=1)

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    print(f"\n{model_name} — Real-World Classification Report\n")
    print(report)

    results_summary[model_name] = {
        "accuracy": acc,
        "y_pred":   y_pred
    }

    safe_name = model_name.lower().replace(" ", "_")

    with open(f"results/real_world_test/{safe_name}_report.txt", "w") as f:
        f.write(f"{model_name} — Real-World Classification Report\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f"{model_name} — Real-World Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/real_world_test/{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")


# ==========================================
# Summary comparison bar chart
# ==========================================

model_names = list(results_summary.keys())
accuracies  = [results_summary[m]["accuracy"] for m in model_names]
colors      = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(model_names, accuracies, color=colors, alpha=0.88, width=0.5)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height()*100:.1f}%",
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )

ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Real-World Test Accuracy — All Models", fontsize=13)
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("results/real_world_test/accuracy_comparison.png", dpi=150)
plt.close()


# ==========================================
# Final summary
# ==========================================

print("\n" + "="*50)
print("REAL-WORLD TEST — FINAL SUMMARY")
print("="*50)
print(f"{'Model':<20} {'TrashNet Val':>14} {'Real-World':>12}")
print("-"*50)

trashnet_acc = {
    "CNN Baseline":              0.5644,
    "MobileNetV2":               0.8713,
    "EfficientNetB0":            0.8990,
    "EfficientNetB0 + Augment": 0.8970,
    "EfficientNetB0 + TACO":    0.9287,
    "EfficientNetB0 + RealWaste": 0.9129,
}

for m in model_names:
    rw  = results_summary[m]["accuracy"]
    tn  = trashnet_acc[m]
    diff = rw - tn
    sign = "+" if diff >= 0 else ""
    print(f"{m:<20} {tn*100:>12.2f}%  {rw*100:>9.2f}%  ({sign}{diff*100:.2f}pp)")

print("\nAll reports and confusion matrices saved to results/real_world_test/")
