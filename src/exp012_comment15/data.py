"""TrashNet datasets for the EXP-012 benchmark.

Images come only from the frozen manifest path lists. This module does not
call image_dataset_from_directory and does not rebuild the split.
"""

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.exp012_comment15.safety import SHUFFLE_BUFFER_SIZE, STUDY_BATCH_SIZES, TRAINING_SEED
from src.seed_study.protocol import CLASS_NAMES, IMAGE_SIZE
from src.seed_study.split import resolved_path


def class_index(path):
    name = path.replace("\\", "/").split("/")[-2]
    if name not in CLASS_NAMES:
        raise ValueError(f"Path is outside the six TrashNet classes: {path}")
    return CLASS_NAMES.index(name)


def make_dataset(paths, *, training, batch_size, training_seed):
    if training:
        if training_seed != TRAINING_SEED:
            raise ValueError(
                f"Training seed must be {TRAINING_SEED}, received {training_seed}"
            )
    elif training_seed is not None:
        raise ValueError("Validation order is fixed. Do not pass a training seed.")
    if batch_size not in STUDY_BATCH_SIZES:
        raise ValueError(
            f"Batch size must be one of {STUDY_BATCH_SIZES}, received {batch_size}"
        )

    absolute = [str(resolved_path(path)) for path in paths]
    labels = [class_index(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices((absolute, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE,
            seed=TRAINING_SEED,
            reshuffle_each_iteration=True,
        )
    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        lambda path, label: (_load_resized_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.map(
        lambda images, batch_labels: (preprocess_input(images), batch_labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def _load_resized_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    return image
