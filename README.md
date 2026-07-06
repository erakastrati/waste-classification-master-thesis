# Waste Classification — Master Thesis

Web app and ML pipeline for classifying household waste into 6 categories:
**cardboard, glass, metal, paper, plastic, trash**.

Current best model: **EfficientNet + Garbage v2 fine-tune (EXP-011)** with Test-Time Augmentation (TTA).  
Real-world accuracy on 600 held-out photos: **92.2%**.

---

## Requirements

| Tool | Version |
|------|---------|
| Python | **3.9** (recommended — TensorFlow 2.16 works best here) |
| pip | latest |

> **Note:** The trained model file is required and is **not** included in git (too large).  
> You need `results/efficientnet_garbage/efficientnet_garbage_best.keras` (~19 MB) on your machine.

---

## Setup (first time only)

Open a terminal and go to the project folder:

```bash
cd waste-classification-master-thesis
```

### 1. Create and activate virtual environment

```bash
python3.9 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

For the **web app only** (minimal):

```bash
pip install -r requirements-app.txt
```

For **full project** (training, evaluation, notebooks):

```bash
pip install -r requirements.txt
pip install flask               # Flask is needed for app.py
```

---

## Run the web app

Make sure you are inside the project folder and the virtual environment is active.

```bash
source venv/bin/activate
python app.py
```

Or without activating venv:

```bash
./venv/bin/python app.py
```

You should see:

```
Loading model...
Model loaded.
 * Running on http://127.0.0.1:5000
```

Open in your browser: **http://127.0.0.1:5000**

Upload a photo of waste → the app returns the predicted class, confidence score, and a recycling tip.

Press `Ctrl + C` in the terminal to stop the server.

---

## How the app works

| File | Role |
|------|------|
| `app.py` | Flask backend — loads model, serves UI, `/predict` endpoint |
| `templates/index.html` | Frontend UI |
| `src/inference/tta_predict.py` | TTA inference (5 views averaged) |
| `results/efficientnet_garbage/efficientnet_garbage_best.keras` | Trained model weights |

On startup the model loads once into memory (~10–20 s on first run). Each prediction takes a few seconds because TTA runs 5 forward passes.

---

## Project structure (main folders)

```
waste-classification-master-thesis/
├── app.py                          # Web app entry point
├── requirements-app.txt            # Minimal deps for the app
├── requirements.txt                # Full project deps
├── templates/index.html            # Web UI
├── src/
│   ├── data/                       # Dataset preparation scripts
│   ├── training/                   # Model training scripts
│   ├── inference/                  # TTA prediction
│   └── evaluation/                 # Real-world evaluation scripts
├── data/
│   ├── real_test_dataset/          # Held-out real photos (600, never used for training)
│   └── garbage-v2-prepared/        # Training data
└── results/
    └── efficientnet_garbage/       # Best model + training logs
```

---

## Real test dataset

Location: `data/real_test_dataset/`

```
data/real_test_dataset/
├── ✅cardboard/   → cardboard1.jpg … cardboard100
├── ✅glass/       → glass1.png … glass100
├── ✅metal/       → metal1.jpg … metal100
├── ✅paper/       → paper1.jpeg … paper100
├── ✅plastic/     → plastic1.png … plastic100
└── ✅trash/       → trash1.jpg … trash100
```

100 photos per class, 600 total. This set is **never used during training** — only for final evaluation.

---

## Evaluate on real test set

```bash
source venv/bin/activate
python -m src.evaluation.evaluate_tta
```

> Evaluation scripts expect class folders named `cardboard`, `glass`, etc.  
> If your folders use the `✅cardboard` prefix, run evaluation with the helper script or rename folders.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'flask'` | `pip install flask` |
| `No module named 'tensorflow'` | `pip install -r requirements-app.txt` |
| Model file not found | Ensure `results/efficientnet_garbage/efficientnet_garbage_best.keras` exists |
| App slow on first prediction | Normal — TensorFlow warms up on first inference |
| Port 5000 already in use | Change port in `app.py`: `app.run(port=5001)` |

---

## Classes

| Class | Icon | Description |
|-------|------|-------------|
| cardboard | 📦 | Cardboard boxes, packaging |
| glass | 🍶 | Glass bottles and jars |
| metal | 🥫 | Cans, metal containers |
| paper | 📄 | Paper, newspapers |
| plastic | ♻️ | Plastic bottles, packaging |
| trash | 🗑️ | Non-recyclable / mixed waste |
