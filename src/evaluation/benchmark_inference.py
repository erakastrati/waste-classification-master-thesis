"""
benchmark_inference.py  —  Inference time / model size benchmark

Measures single-image inference latency (CPU, matching typical deployment)
for the models compared in the thesis, plus model file size and parameter
count. Used to fill the "efficiency" gap in Chapter 4 (MobileNetV2 vs
EfficientNetB0 speed/size trade-off) and to support Recommendation #2
(mobile deployment).

Run from project root:
    ./venv/bin/python -m src.evaluation.benchmark_inference
"""

import os
import platform
import subprocess
import time
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

IMAGE_SIZE = (224, 224)
N_WARMUP = 10
N_RUNS = 100

MODELS = {
    "CNN Baseline": {
        "path": "results/cnn_baseline/cnn_baseline.keras",
        "preprocess": lambda x: x / 255.0,
    },
    "MobileNetV2": {
        "path": "results/mobilenet/mobilenet_best.keras",
        "preprocess": mobilenet_preprocess,
    },
    "EfficientNetB0 (final)": {
        "path": "results/efficientnet_garbage/efficientnet_garbage_best.keras",
        "preprocess": efficientnet_preprocess,
    },
}

# Use a real sample image so preprocessing behaves exactly as in production
SAMPLE_IMAGE = "data/trashNet-dataset/cardboard/cardboard1.jpg"


def get_dummy_image():
    if os.path.exists(SAMPLE_IMAGE):
        img = Image.open(SAMPLE_IMAGE).convert("RGB").resize(IMAGE_SIZE)
        return np.array(img, dtype=np.float32)
    return np.random.rand(*IMAGE_SIZE, 3).astype(np.float32) * 255.0


def benchmark_model(name, path, preprocess_fn):
    print(f"\nLoading {name}...")
    model = load_model(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    params = model.count_params()

    img = get_dummy_image()
    batch = np.expand_dims(preprocess_fn(img.copy()), axis=0)

    for _ in range(N_WARMUP):
        model.predict(batch, verbose=0)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        model.predict(batch, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "name": name,
        "size_mb": size_mb,
        "params": params,
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "median_ms": np.median(times),
    }


def get_cpu_label():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return platform.processor() or platform.machine()


if __name__ == "__main__":
    cpu_label = get_cpu_label()
    print(f"CPU: {cpu_label}")
    print(f"TensorFlow: {tf.__version__}  |  GPUs visible: {tf.config.list_physical_devices('GPU')}")

    results = []
    for name, cfg in MODELS.items():
        r = benchmark_model(name, cfg["path"], cfg["preprocess"])
        results.append(r)

    print("\n" + "=" * 82)
    print(f"{'Model':<25}{'Params':>12}{'Size (MB)':>12}{'Mean (ms)':>12}{'Std (ms)':>10}{'Median (ms)':>13}")
    print("=" * 82)
    for r in results:
        print(f"{r['name']:<25}{r['params']:>12,}{r['size_mb']:>12.2f}{r['mean_ms']:>12.2f}{r['std_ms']:>10.2f}{r['median_ms']:>13.2f}")

    # TTA = 5x single-pass inference calls (matches tta_predict.py: 5 views averaged)
    eff = next(r for r in results if r["name"] == "EfficientNetB0 (final)")
    print(f"\nEfficientNetB0 + TTA (5 views), estimated: {eff['mean_ms'] * 5:.2f} ms/image "
          f"(std ~{eff['std_ms'] * 5:.2f} ms)")

    os.makedirs("results/benchmark", exist_ok=True)
    with open("results/benchmark/inference_benchmark.txt", "w") as f:
        f.write("Inference time / model size benchmark (CPU only, no GPU, single image, batch=1)\n")
        f.write(f"CPU: {cpu_label}\n")
        f.write(f"TensorFlow: {tf.__version__}  |  GPUs visible: {tf.config.list_physical_devices('GPU')}\n")
        f.write(f"N_RUNS={N_RUNS}, N_WARMUP={N_WARMUP}, image_size={IMAGE_SIZE}\n\n")
        f.write(f"{'Model':<25}{'Params':>12}{'Size (MB)':>12}{'Mean (ms)':>12}{'Std (ms)':>10}{'Median (ms)':>13}\n")
        for r in results:
            f.write(f"{r['name']:<25}{r['params']:>12,}{r['size_mb']:>12.2f}{r['mean_ms']:>12.2f}{r['std_ms']:>10.2f}{r['median_ms']:>13.2f}\n")
        f.write(f"\nEfficientNetB0 + TTA (5 views), estimated: {eff['mean_ms'] * 5:.2f} ms/image "
                f"(std ~{eff['std_ms'] * 5:.2f} ms)\n")

    print("\nSaved to results/benchmark/inference_benchmark.txt")
