from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input


def build_cnn_model():

    model = Sequential([

        Input(shape=(224, 224, 3)),

        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        GlobalAveragePooling2D(),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(6, activation="softmax")
    ])

    return model