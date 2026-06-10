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
Training pending

Results:
To be completed after model training.

EXP-001 Results

Epochs:
10

Train Accuracy:
54.65%

Validation Accuracy:
51.88%

Train Loss:
1.1716

Validation Loss:
1.1766

Observations:

- The model learned meaningful visual patterns.
- No significant overfitting was observed.
- Performance remains limited, suggesting that more advanced architectures may achieve better results.


EXP-001 Results

Architecture:
CNN Baseline

Total Parameters:
110,534

Training Accuracy:
54.90%

Validation Accuracy:
50.30%

Training Loss:
1.15

Validation Loss:
1.27

Observations:

- Stable convergence observed.
- No severe overfitting detected.
- Validation accuracy remained around 50%.
- Baseline performance leaves room for improvement using transfer learning architectures.