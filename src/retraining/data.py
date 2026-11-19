"""Manifest loader for the corrected TrashNet split.

Does not call image_dataset_from_directory and does not draw a new split.
"""

import random
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.retraining.protocol import (
    BATCH_SIZE,
    CLASS_TO_INDEX,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_TRAIN_N,
    EXPECTED_VAL_COUNTS,
    EXPECTED_VAL_N,
    IMAGE_SIZE,
    PROJECT_ROOT,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)


def read_manifest(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    labels = []
    for relative in lines:
        class_name = relative.split("/")[2]
        if class_name not in CLASS_TO_INDEX:
            raise ValueError(f"Path is outside the six-class table: {relative}")
        if not (PROJECT_ROOT / relative).is_file():
            raise ValueError(f"Manifest path is missing on disk: {relative}")
        labels.append(CLASS_TO_INDEX[class_name])
    return lines, labels


def load_split():
    train_paths, train_labels = read_manifest(TRAIN_MANIFEST)
    val_paths, val_labels = read_manifest(VAL_MANIFEST)
    if len(train_paths) != EXPECTED_TRAIN_N or len(val_paths) != EXPECTED_VAL_N:
        raise ValueError(
            f"Manifest sizes are {len(train_paths)} and {len(val_paths)}, "
            f"expected {EXPECTED_TRAIN_N} and {EXPECTED_VAL_N}"
        )
    if train_paths != sorted(train_paths) or val_paths != sorted(val_paths):
        raise ValueError("Manifests are not in sorted path order")
    return train_paths, train_labels, val_paths, val_labels


def class_counts(paths):
    return dict(Counter(path.split("/")[2] for path in paths))


def assert_expected_counts(train_paths, val_paths):
    if class_counts(train_paths) != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"Training counts {class_counts(train_paths)}")
    if class_counts(val_paths) != EXPECTED_VAL_COUNTS:
        raise ValueError(f"Validation counts {class_counts(val_paths)}")


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = preprocess_input(image)
    image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return image


def load_batch(paths, labels):
    absolute = [str(PROJECT_ROOT / relative) for relative in paths]
    images = np.stack([load_image(path).numpy() for path in absolute])
    targets = np.asarray(labels, dtype=np.int32)
    return images, targets


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


def training_dataset(paths, labels, training_seed):
    """Infinite dataset. Each epoch is one full shuffle from random.Random(seed)."""

    def generator():
        order = list(range(len(paths)))
        rng = random.Random(training_seed)
        while True:
            rng.shuffle(order)
            for start in range(0, len(order), BATCH_SIZE):
                chosen = order[start:start + BATCH_SIZE]
                batch_paths = [paths[index] for index in chosen]
                batch_labels = [labels[index] for index in chosen]
                yield load_batch(batch_paths, batch_labels)

    return _dataset_from_generator(generator)


def validation_dataset(paths, labels):
    """Finite pass in manifest order, then repeated for later epochs."""

    def generator():
        for start in range(0, len(paths), BATCH_SIZE):
            yield load_batch(
                paths[start:start + BATCH_SIZE],
                labels[start:start + BATCH_SIZE],
            )

    return _dataset_from_generator(generator).repeat()


def steps_per_epoch(n_images):
    return (n_images + BATCH_SIZE - 1) // BATCH_SIZE
