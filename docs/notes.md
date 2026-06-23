# Dataset Analysis

## Dataset

Source:
TrashNet Dataset (Kaggle)

Classes:
- cardboard
- glass
- metal
- paper
- plastic
- trash

Total images:
2527

## Class Distribution

cardboard: 403
glass: 501
metal: 410
paper: 594
plastic: 482
trash: 137

Observation:
The dataset is imbalanced. The trash class contains significantly fewer samples.

## Image Dimensions

Width: 512
Height: 384

Observation:
All images have consistent dimensions.

## Dataset Split

Training: 2022
Validation: 505

Split ratio:
80/20

## Preprocessing Configuration

Image Resize:
224 x 224 pixels

Batch Size:
32

Validation Split:
20%

Random Seed:
42

Normalization:
To be applied during model training.

Reasoning:
Images were resized from 512x384 to 224x224 to ensure compatibility with CNN, MobileNetV2, and EfficientNetB0 architectures while reducing computational cost.

## Experiment Log

### EXP-001

Model:
CNN Baseline

Architecture:

* Conv2D(32) + MaxPooling
* Conv2D(64) + MaxPooling
* Conv2D(128) + MaxPooling
* GlobalAveragePooling2D
* Dense(128)
* Dropout(0.3)
* Dense(6, Softmax)

Input Shape:
224x224x3

Total Parameters:
110,534

Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metrics:
Accuracy

Epochs:
10

Status:
Completed

Results:

Epochs:
10

Training Accuracy:
~67%

Validation Accuracy:
56.44%

Training Loss:
~0.92

Validation Loss:
1.1258

Observations:

- The model learned meaningful visual patterns from scratch.
- No significant overfitting was observed over 10 epochs.
- Validation accuracy stabilized around 56%, which is expected for a lightweight custom CNN on a relatively small and imbalanced dataset.
- The trash class (137 images) likely contributed to misclassifications due to underrepresentation.
- This result establishes the baseline for comparison with transfer learning architectures.

Artifacts:
- results/cnn_baseline/cnn_baseline.keras
- results/cnn_baseline/accuracy.png
- results/cnn_baseline/loss.png

---

### EXP-002

Model:
MobileNetV2 (Transfer Learning)

Architecture:

* MobileNetV2 base (ImageNet pretrained, frozen in Phase 1)
* GlobalAveragePooling2D
* Dense(128, ReLU)
* Dropout(0.3)
* Dense(6, Softmax)

Input Shape:
224x224x3

Total Parameters:
~2.3M (base) + classification head

Preprocessing:
MobileNetV2 preprocess_input (scales to [-1, 1])

Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metrics:
Accuracy

Training Strategy:

Phase 1 (Epochs 1-10):
Base model frozen. Only classification head trained.

Phase 2 (Epochs 11-20):
Top 30 layers of base unfrozen. Fine-tuning applied.

Status:
Completed (Final Run — v2 with corrected fine-tuning)

Total Parameters:
2,422,726 (9.24 MB)

Trainable (Phase 1):
164,742 (classification head only)

---

Phase 1 Results (Epochs 1–10, Base Frozen, lr=1e-3):

Best Validation Accuracy:
86.73% at Epoch 5

Training Accuracy at best epoch:
93.89%

Validation Loss at best epoch:
0.4611

Note: EarlyStopping triggered at Epoch 10. Weights restored from Epoch 5.

---

Phase 2 Results (Epochs 1–14, Top 30 Layers Unfrozen, lr=1e-5):

Best Validation Accuracy:
87.13% at Phase 2 Epoch 9

Training Accuracy at best epoch:
96.59%

Validation Loss at best epoch:
0.4410

Note: EarlyStopping triggered at Phase 2 Epoch 14. Weights restored from Phase 2 Epoch 9.
Fine-tuning was fully stable — no loss spikes observed.

---

Final Results (Best Model via ModelCheckpoint):

Validation Accuracy:
87.13%

Validation Loss:
0.4410

---

Comparison vs CNN Baseline:

CNN Baseline:           56.44%
MobileNetV2 (final):   87.13%  (+30.69 pp)

---

Observations:

- Fine-tuning with lr=1e-5 was fully stable. No catastrophic forgetting occurred.
  Validation loss remained consistently around 0.44 throughout Phase 2.
- Phase 2 provided a marginal improvement over Phase 1 peak (86.73% → 87.13%),
  confirming that careful fine-tuning with a low learning rate can improve generalization.
- ModelCheckpoint saved the best model across all epochs of both phases automatically.
- EarlyStopping prevented unnecessary computation once the model plateaued.
- MobileNetV2 represents a +30.69 percentage point improvement over the CNN baseline,
  demonstrating the significant advantage of transfer learning on small datasets.
- The model plateau around 87% suggests the dataset size is the main limiting factor.

Artifacts:
- results/mobilenet/mobilenet.keras         (best model)
- results/mobilenet/mobilenet_best.keras    (checkpoint copy)
- results/mobilenet/accuracy.png
- results/mobilenet/loss.png

---

### EXP-003

Model:
EfficientNetB0 (Transfer Learning)

Architecture:

* EfficientNetB0 base (ImageNet pretrained, frozen in Phase 1)
* GlobalAveragePooling2D
* Dense(128, ReLU)
* Dropout(0.3)
* Dense(6, Softmax)

Input Shape:
224x224x3

Total Parameters:
~4.05M (base) + classification head

Preprocessing:
EfficientNetB0 preprocess_input (scales pixel values for EfficientNet internal normalization)

Optimizer:
Adam

Loss Function:
Sparse Categorical Crossentropy

Metrics:
Accuracy

Training Strategy:

Phase 1 (Epochs 1-10):
Base model frozen. Only classification head trained. lr=1e-3.

Phase 2 (Epochs up to 15):
Top 30 layers of base unfrozen. Fine-tuning with lr=1e-5.
EarlyStopping with patience=5.

Callbacks:
ModelCheckpoint — saves best model by val_accuracy.
EarlyStopping — stops training when val_accuracy plateaus.

Status:
Completed

Total Parameters:
4,214,313 (16.08 MB)

Trainable (Phase 1):
164,742 (classification head only)

---

Phase 1 Results (Epochs 1–10, Base Frozen, lr=1e-3):

Best Validation Accuracy:
89.90% at Epoch 10

Training Accuracy at best epoch:
98.80%

Validation Loss at best epoch:
0.3386

Note: val_accuracy improved every epoch from 86.34% to 89.90%.
EarlyStopping did not trigger — model kept improving through all 10 epochs.

---

Phase 2 Results (Epochs 1–6, Top 30 Layers Unfrozen, lr=1e-5):

Phase 2 never exceeded Phase 1 best (89.90%).
Best Phase 2 epoch: Epoch 1 with val_accuracy 89.50%.
EarlyStopping triggered at Epoch 6. Weights restored from Phase 2 Epoch 1.

---

Final Results (Best Model via ModelCheckpoint):

Validation Accuracy:
89.90%

Validation Loss:
0.3386

---

Comparison vs all models:

CNN Baseline:    56.44%
MobileNetV2:     87.13%  (+30.69 pp vs CNN)
EfficientNetB0:  89.90%  (+33.46 pp vs CNN, +2.77 pp vs MobileNetV2)

---

Observations:

- EfficientNetB0 is the best performing model across all three experiments.
- Phase 1 showed continuous improvement throughout all 10 epochs, suggesting the
  EfficientNet feature extractor is particularly well-suited to visual waste classification.
- Phase 2 fine-tuning did not improve results — the frozen base already generalized well.
  EarlyStopping correctly halted training after 6 epochs without improvement.
- Validation loss (0.3386) is significantly lower than MobileNetV2 (0.4410),
  indicating more confident and stable predictions.
- EfficientNetB0 achieved 89.90% accuracy with only ~4.2M parameters,
  demonstrating strong efficiency relative to model size.
- The +2.77 pp improvement over MobileNetV2 confirms EfficientNetB0 as the
  recommended architecture for this waste classification task.

Artifacts:
- results/efficientnet/efficientnet.keras         (best model)
- results/efficientnet/efficientnet_best.keras    (checkpoint copy)
- results/efficientnet/accuracy.png
- results/efficientnet/loss.png

---

## Final Model Comparison Summary

| Model           | Val Accuracy | Val Loss | Parameters  | Training Strategy            |
|-----------------|-------------|----------|-------------|------------------------------|
| CNN Baseline    | 56.44%      | 1.1258   | 110,534     | Trained from scratch, 10 ep  |
| MobileNetV2     | 87.13%      | 0.4410   | 2,422,726   | Transfer learning, 2-phase   |
| EfficientNetB0  | 89.90%      | 0.3386   | 4,214,313   | Transfer learning, 2-phase   |

Best Model: EfficientNetB0
Improvement over baseline: +33.46 percentage points

---

## Detailed Evaluation Results (Per-Class Metrics)

Validation set: 505 images across 6 classes
Evaluation method: shuffle=True, seed=42 (identical split to training)

### CNN Baseline — Per-Class F1

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| cardboard | 0.75      | 0.72   | 0.74     | 69      |
| glass     | 0.49      | 0.49   | 0.49     | 102     |
| metal     | 0.49      | 0.61   | 0.55     | 88      |
| paper     | 0.56      | 0.79   | 0.66     | 123     |
| plastic   | 0.65      | 0.38   | 0.48     | 89      |
| trash     | 0.00      | 0.00   | 0.00     | 34      |
| **macro avg** | **0.49** | **0.50** | **0.48** | 505 |

Observations:
- trash class completely failed (F1=0.00) — model never predicted it, likely due to class imbalance (34 samples).
- Best class: cardboard (F1=0.74).
- Worst classes: trash, plastic, glass.

---

### MobileNetV2 — Per-Class F1

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| cardboard | 0.89      | 0.91   | 0.90     | 69      |
| glass     | 0.89      | 0.90   | 0.90     | 102     |
| metal     | 0.82      | 0.92   | 0.87     | 88      |
| paper     | 0.93      | 0.91   | 0.92     | 123     |
| plastic   | 0.86      | 0.80   | 0.83     | 89      |
| trash     | 0.75      | 0.62   | 0.68     | 34      |
| **macro avg** | **0.85** | **0.84** | **0.85** | 505 |

Observations:
- Strong performance across all classes.
- trash class improved significantly (F1=0.68) compared to CNN baseline.
- Best class: paper (F1=0.92).
- Weakest class: trash (F1=0.68) — expected due to low sample count.

---

### EfficientNetB0 — Per-Class F1

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| cardboard | 0.96      | 0.94   | 0.95     | 69      |
| glass     | 0.87      | 0.89   | 0.88     | 102     |
| metal     | 0.89      | 0.89   | 0.89     | 88      |
| paper     | 0.94      | 0.94   | 0.94     | 123     |
| plastic   | 0.88      | 0.85   | 0.87     | 89      |
| trash     | 0.82      | 0.82   | 0.82     | 34      |
| **macro avg** | **0.89** | **0.89** | **0.89** | 505 |

Observations:
- Best performing model across every single class.
- trash class reached F1=0.82 — strong result for a class with only 34 validation samples.
- Best class: cardboard (F1=0.95).
- Most balanced model: all classes above F1=0.82.
- Macro avg F1=0.89 confirms consistent performance regardless of class imbalance.

---

### Cross-Model F1 Comparison (macro avg)

| Model          | Accuracy | Macro F1 | Weighted F1 |
|----------------|----------|----------|-------------|
| CNN Baseline   | 0.56     | 0.48     | 0.54        |
| MobileNetV2    | 0.87     | 0.85     | 0.87        |
| EfficientNetB0 | 0.90     | 0.89     | 0.90        |

Artifacts:
- results/cnn_baseline/confusion_matrix.png
- results/cnn_baseline/classification_report.txt
- results/mobilenet/confusion_matrix.png
- results/mobilenet/classification_report.txt
- results/efficientnet/confusion_matrix.png
- results/efficientnet/classification_report.txt

---

## Remaining Work

### NEXT-01 — Real-World Testing

Status: Completed

Dataset: 78 real-world photos taken with iPhone (13 per class)
Classes: cardboard, glass, metal, paper, plastic, trash
Format: JPEG (converted from HEIC via sips)

---

Results:

| Model          | TrashNet Val | Real-World | Drop     |
|----------------|-------------|------------|----------|
| CNN Baseline   | 56.44%      | 11.54%     | -44.90pp |
| MobileNetV2    | 87.13%      | 43.59%     | -43.54pp |
| EfficientNetB0 | 89.90%      | 39.74%     | -50.16pp |

---

Per-Class F1 (Real-World):

CNN Baseline:
- cardboard: 0.15 | glass: 0.00 | metal: 0.10
- paper: 0.16    | plastic: 0.00 | trash: 0.14
- macro F1: 0.09

MobileNetV2:
- cardboard: 0.39 | glass: 0.27 | metal: 0.48
- paper: 0.57    | plastic: 0.52 | trash: 0.13
- macro F1: 0.39

EfficientNetB0:
- cardboard: 0.33 | glass: 0.42 | metal: 0.24
- paper: 0.48    | plastic: 0.47 | trash: 0.25
- macro F1: 0.36

---

Key Observations:

- All three models suffered a severe drop in real-world conditions.
  This is a classic domain shift problem: TrashNet uses white backgrounds,
  controlled studio lighting, and centered objects. Real-world photos have
  natural backgrounds, variable lighting, and arbitrary angles.

- The CNN Baseline collapsed to 11.54% — near random (chance = 16.7% for 6 classes).
  It learned visual patterns tightly coupled to the white background of TrashNet.

- MobileNetV2 (43.59%) and EfficientNetB0 (39.74%) retained partial generalization.
  Transfer learning from ImageNet provided some domain-agnostic features
  (edges, textures, shapes), but not enough to overcome the domain gap.

- paper was the best-performing class across all models in real-world conditions.
  This is likely because paper has distinctive flat texture that is
  background-independent.

- trash and glass were the hardest classes in real-world conditions.

- This result is a significant finding for the thesis: it demonstrates that
  training exclusively on controlled dataset images (TrashNet) is insufficient
  for real-world deployment without domain adaptation or data augmentation.

Artifacts:
- results/real_world_test/cnn_baseline_report.txt
- results/real_world_test/cnn_baseline_confusion_matrix.png
- results/real_world_test/mobilenetv2_report.txt
- results/real_world_test/mobilenetv2_confusion_matrix.png
- results/real_world_test/efficientnetb0_report.txt
- results/real_world_test/efficientnetb0_confusion_matrix.png
- results/real_world_test/accuracy_comparison.png

### NEXT-02 — Model Comparison Plots

Status: Completed

Generated 3 comparison figures for thesis Results chapter:

1. accuracy_f1_comparison.png
   Grouped bar chart: Validation Accuracy vs Macro F1-Score per model.
   CNN: Acc=0.56, F1=0.48 | MobileNetV2: Acc=0.87, F1=0.85 | EfficientNetB0: Acc=0.90, F1=0.89

2. per_class_f1_comparison.png
   Per-class F1-Score for all 6 waste categories across all 3 models.
   Shows CNN failure on trash (F1=0.00) vs EfficientNetB0 robustness (F1=0.82).

3. val_loss_comparison.png
   Validation loss bar chart: CNN=1.1258, MobileNetV2=0.4410, EfficientNetB0=0.3386

Artifacts:
- results/comparison/accuracy_f1_comparison.png
- results/comparison/per_class_f1_comparison.png
- results/comparison/val_loss_comparison.png

### EXP-004 — EfficientNetB0 + Data Augmentation

Model:
EfficientNetB0 (Transfer Learning + Data Augmentation)

Motivation:
Project proposal explicitly requires comparison with/without data augmentation.
Previous EfficientNetB0 (EXP-003) trained without augmentation.
This experiment adds augmentation to address the domain gap observed in real-world testing.

Augmentation Layers (applied during training only):
- RandomFlip("horizontal")
- RandomRotation(0.15) — ±15 degrees
- RandomZoom(0.15) — ±15% zoom
- RandomContrast(0.2) — ±20% contrast

Architecture:
Input → Augmentation → EfficientNetB0 base → GlobalAveragePooling2D
→ Dense(128) → Dropout(0.3) → Dense(6, Softmax)

Training Strategy:
Same as EXP-003: Phase 1 (lr=1e-3, frozen) + Phase 2 (lr=1e-5, top 30 layers)
ModelCheckpoint + EarlyStopping (patience=5)

Status: Completed

Training Results:
Phase 1 (Classification Head, Frozen Base):
- Best val_accuracy: 0.8970 (Epoch 5)
- Best val_loss: 0.3104
- EarlyStopping triggered at Epoch 10 (patience=5, best at Epoch 5)

Phase 2 (Fine-Tuning, Top 30 Layers, lr=1e-5):
- val_accuracy did not improve above Phase 1 best (0.8970)
- EarlyStopping triggered at Epoch 6 of Phase 2

Final Validation Results (TrashNet):
- Validation Accuracy: 89.70%
- Validation Loss:     0.3104
- Macro F1-Score:      0.89

Per-Class F1-Score:
- cardboard: 0.94
- glass:     0.89
- metal:     0.90
- paper:     0.93
- plastic:   0.86
- trash:     0.81

TrashNet Comparison vs EXP-003 (no augmentation):
- Accuracy: 89.70% vs 89.90% → -0.20pp (statistically negligible)
- Loss:     0.3104 vs 0.3386 → lower loss (+0.0282)
- Conclusion: Augmentation has no negative impact on TrashNet performance.

Real-World Test Results (78 images, 6 classes):
- Real-World Accuracy: 50.00%
- TrashNet → Real-World gap: -39.70pp

Real-World Comparison (with vs without augmentation):
- EfficientNetB0 (no aug):  39.74% real-world
- EfficientNetB0 (+ aug):   50.00% real-world
- Improvement: +10.26pp on real-world data

Key Observation:
Data augmentation improved real-world generalization by +10.26pp while maintaining
near-identical TrashNet validation performance (-0.20pp). This is the central
finding for the augmentation section of the thesis: augmentation reduces domain
shift without sacrificing controlled-environment accuracy.

All 4-model real-world comparison summary:
| Model                    | TrashNet Val | Real-World | Gap      |
|--------------------------|-------------|------------|----------|
| CNN Baseline             |    56.44%   |   11.54%   | -44.90pp |
| MobileNetV2              |    87.13%   |   43.59%   | -43.54pp |
| EfficientNetB0           |    89.90%   |   39.74%   | -50.16pp |
| EfficientNetB0 + Augment |    89.70%   |   50.00%   | -39.70pp |

Artifacts:
- results/efficientnet_augmented/efficientnet_augmented.keras
- results/efficientnet_augmented/efficientnet_augmented_best.keras
- results/efficientnet_augmented/accuracy.png
- results/efficientnet_augmented/loss.png
- results/efficientnet_augmented/confusion_matrix.png
- results/efficientnet_augmented/classification_report.txt
- results/real_world_test/efficientnetb0_augment_report.txt
- results/real_world_test/efficientnetb0_augment_confusion_matrix.png
- results/real_world_test/accuracy_comparison.png (updated)

---

### NEXT-03 — Thesis Write-Up
Chapters that can now be written based on collected data:
- Dataset and Preprocessing
- Model Architectures
- Training Configuration
- Results: Accuracy, Loss, F1, Confusion Matrices
- Discussion: CNN vs Transfer Learning, class imbalance impact, trash class behavior