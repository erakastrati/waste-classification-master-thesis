import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("results/comparison", exist_ok=True)


# ==========================================
# Metrics from evaluation results
# ==========================================

models = ["CNN Baseline", "MobileNetV2", "EfficientNetB0"]

accuracy     = [0.5644, 0.8713, 0.8990]
macro_f1     = [0.48,   0.85,   0.89]
val_loss     = [1.1258, 0.4410, 0.3386]

classes      = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

f1_per_class = {
    "CNN Baseline":   [0.74, 0.49, 0.55, 0.66, 0.48, 0.00],
    "MobileNetV2":    [0.90, 0.90, 0.87, 0.92, 0.83, 0.68],
    "EfficientNetB0": [0.95, 0.88, 0.89, 0.94, 0.87, 0.82],
}

colors = ["#4C72B0", "#DD8452", "#55A868"]


# ==========================================
# Plot 1: Accuracy and Macro F1 Bar Chart
# ==========================================

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

bars1 = ax.bar(x - width / 2, accuracy, width, label="Validation Accuracy", color=colors, alpha=0.9)
bars2 = ax.bar(x + width / 2, macro_f1,  width, label="Macro F1-Score",      color=colors, alpha=0.55)

for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.2f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{bar.get_height():.2f}",
        ha="center", va="bottom", fontsize=10
    )

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison — Accuracy and Macro F1-Score", fontsize=13)
ax.set_ylim(0, 1.12)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("results/comparison/accuracy_f1_comparison.png", dpi=150)
plt.close()


# ==========================================
# Plot 2: Per-Class F1 Score Comparison
# ==========================================

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

for i, (model, color) in enumerate(zip(models, colors)):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, f1_per_class[model], width, label=model, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_ylabel("F1-Score", fontsize=11)
ax.set_title("Per-Class F1-Score Comparison", fontsize=13)
ax.set_ylim(0, 1.1)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("results/comparison/per_class_f1_comparison.png", dpi=150)
plt.close()


# ==========================================
# Plot 3: Validation Loss Comparison
# ==========================================

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(models, val_loss, color=colors, alpha=0.88, width=0.5)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.4f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_ylabel("Validation Loss", fontsize=11)
ax.set_title("Model Comparison — Validation Loss", fontsize=13)
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("results/comparison/val_loss_comparison.png", dpi=150)
plt.close()


print("Saved: results/comparison/accuracy_f1_comparison.png")
print("Saved: results/comparison/per_class_f1_comparison.png")
print("Saved: results/comparison/val_loss_comparison.png")
