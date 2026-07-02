"""
ensemble_predict.py  —  Multi-model ensemble + TTA inference

Averages softmax outputs from Household, TACO, and RealWaste fine-tuned models,
each evaluated with TTA.

Used by app.py and evaluation scripts.
"""

from tensorflow.keras.models import load_model

from src.inference.tta_predict import predict_with_tta

ENSEMBLE_MODEL_PATHS = [
    "results/efficientnet_household/efficientnet_household.keras",
    "results/efficientnet_taco/efficientnet_taco.keras",
    "results/efficientnet_realwaste/efficientnet_realwaste.keras",
]


def load_ensemble_models(paths=None):
    paths = paths or ENSEMBLE_MODEL_PATHS
    models = []
    for path in paths:
        print(f"  Loading {path}...")
        models.append(load_model(path))
    return models


def predict_ensemble_tta(models, img, image_size=(224, 224)):
    """Average TTA predictions across all models. Returns softmax vector."""
    preds_sum = None
    for model in models:
        pred = predict_with_tta(model, img, image_size)
        preds_sum = pred if preds_sum is None else preds_sum + pred
    return preds_sum / len(models)
