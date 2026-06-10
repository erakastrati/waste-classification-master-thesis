import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2


def build_mobilenet_model():

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation="relu")(x)

    x = Dropout(0.3)(x)

    outputs = Dense(6, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model, base_model
