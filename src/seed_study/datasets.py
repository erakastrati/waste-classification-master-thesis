"""Datasets built from the frozen path lists.

The training seed controls only the training-buffer shuffle. Validation order
is the manifest order. This module does not call image_dataset_from_directory.
"""

import tensorflow as tf

from src.seed_study.protocol import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    SHUFFLE_BUFFER_SIZE,
    TRAINING_SEEDS,
)
from src.seed_study.split import resolved_path

from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


def cnn_preprocess(image):
    return image / 255.0


PREPROCESS = {
    "exp001": cnn_preprocess,
    "exp002": mobilenet_preprocess,
    "exp003": efficientnet_preprocess,
}


def labels_for(paths):
    labels = []
    for path in paths:
        name = path.split("/")[-2]
        if name not in CLASS_NAMES:
            raise ValueError(f"Path is outside the six TrashNet classes: {path}")
        labels.append(CLASS_NAMES.index(name))
    return labels


def absolute_paths(paths):
    return [str(resolved_path(path)) for path in paths]


def path_label_dataset(paths, *, training, training_seed):
    """Path/label dataset before image decoding.

    Validation rejects a training seed so evaluation order cannot follow it.
    """
    if training:
        if isinstance(training_seed, bool) or training_seed not in TRAINING_SEEDS:
            raise ValueError(
                f"Training shuffle seed must be one of {TRAINING_SEEDS}, "
                f"received {training_seed}"
            )
    elif training_seed is not None:
        raise ValueError(
            "Validation order is fixed. Do not pass a training seed for validation."
        )
    ds = tf.data.Dataset.from_tensor_slices(
        (absolute_paths(paths), labels_for(paths))
    )
    if training:
        ds = ds.shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE,
            seed=training_seed,
            reshuffle_each_iteration=True,
        )
    options = tf.data.Options()
    options.deterministic = True
    return ds.with_options(options)


def load_resized_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return image


def prepare_dataset(paths, experiment, *, training, training_seed):
    if experiment not in PREPROCESS:
        raise ValueError(f"Unknown experiment: {experiment}")
    preprocess = PREPROCESS[experiment]
    ds = path_label_dataset(paths, training=training, training_seed=training_seed)
    ds = ds.map(
        lambda path, label: (load_resized_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(
        lambda images, labels: (preprocess(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
