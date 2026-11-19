"""Corrected-split loader for EXP-001 and EXP-002.

Paths and labels come from the approved manifest reader. Preprocessing is
applied here and is not the EfficientNet preprocessing used by EXP-003.
"""

import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from src.retraining.data import assert_expected_counts, load_split, steps_per_epoch
from src.corrected_baselines.protocol import BATCH_SIZE, IMAGE_SIZE, PROJECT_ROOT

__all__ = [
    "assert_expected_counts",
    "cnn_preprocess",
    "decode_resized",
    "epoch_orders",
    "load_batch",
    "load_split",
    "mobilenet_preprocess",
    "steps_per_epoch",
    "training_dataset",
    "validation_dataset",
]


def cnn_preprocess(image):
    return image / 255.0


PREPROCESS = {
    "EXP-001": cnn_preprocess,
    "EXP-002": mobilenet_preprocess,
}


def decode_resized(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return image


def load_batch(paths, labels, preprocess):
    absolute = [str(PROJECT_ROOT / relative) for relative in paths]
    images = np.stack([
        preprocess(decode_resized(path)).numpy() for path in absolute
    ])
    targets = np.asarray(labels, dtype=np.int32)
    return images, targets


def epoch_orders(n_images, training_seed):
    """Full permutations from random.Random(training_seed), one per epoch."""
    order = list(range(n_images))
    rng = random.Random(training_seed)
    while True:
        rng.shuffle(order)
        yield list(order)


def _dataset_from_generator(generator):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    options = tf.data.Options()
    options.deterministic = True
    return dataset.with_options(options)


def training_dataset(paths, labels, training_seed, preprocess):
    def generator():
        for order in epoch_orders(len(paths), training_seed):
            for start in range(0, len(order), BATCH_SIZE):
                chosen = order[start:start + BATCH_SIZE]
                yield load_batch(
                    [paths[index] for index in chosen],
                    [labels[index] for index in chosen],
                    preprocess,
                )

    return _dataset_from_generator(generator)


def validation_dataset(paths, labels, preprocess):
    def generator():
        for start in range(0, len(paths), BATCH_SIZE):
            yield load_batch(
                paths[start:start + BATCH_SIZE],
                labels[start:start + BATCH_SIZE],
                preprocess,
            )

    return _dataset_from_generator(generator).repeat()
