"""
tta_predict.py  —  Test-Time Augmentation inference

Averages softmax predictions over multiple views of the same image
(resize, flip, center crops) to reduce sensitivity to background and framing.

Used by app.py and evaluation scripts.
"""

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.efficientnet import preprocess_input


def _center_crop(img, ratio):
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def tta_views(img, image_size=(224, 224)):
    """Return PIL views used for TTA (original + augmentations)."""
    base = img.convert("RGB")
    views = [
        base,
        ImageOps.mirror(base),
        _center_crop(base, 0.85),
        _center_crop(base, 0.70),
        ImageOps.mirror(_center_crop(base, 0.85)),
    ]
    return [v.resize(image_size) for v in views]


def predict_with_tta(model, img, image_size=(224, 224)):
    """
    Run TTA inference on a PIL image.
    Returns averaged softmax vector (shape: num_classes).
    """
    preds_sum = None
    n_views = 0

    for view in tta_views(img, image_size):
        arr = np.array(view, dtype=np.float32)
        arr = np.expand_dims(preprocess_input(arr), axis=0)
        batch_pred = model.predict(arr, verbose=0)[0]
        preds_sum = batch_pred if preds_sum is None else preds_sum + batch_pred
        n_views += 1

    return preds_sum / n_views
