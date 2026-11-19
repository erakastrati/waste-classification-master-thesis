"""Runtime environment recorded with every controlled run."""

import os
import platform
import sys


def environment_record():
    import keras
    import tensorflow as tf

    devices = [
        {"type": device.device_type, "name": device.name}
        for device in tf.config.list_physical_devices()
    ]
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "devices": devices,
        "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
        "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        "op_determinism_enabled": False,
        "tf_data_pipeline_deterministic": True,
        "bit_for_bit_guarantee": False,
        "bit_for_bit_note": (
            "keras.utils.set_random_seed controls Python, NumPy, and TensorFlow "
            "RNGs, and the training shuffle uses the training seed. This does not "
            "guarantee bit-for-bit reproducibility across hardware, TensorFlow "
            "builds, or non-deterministic ops. Operation determinism is left off, "
            "as in the original experiments."
        ),
    }
