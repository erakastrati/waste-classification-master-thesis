"""Train one controlled seed of EXP-001, EXP-002, or EXP-003.

Run a single experiment and a single training seed per process. Outputs go
only to results/seed_study/. Historical checkpoints are not read or written.

Example:
    python -m src.seed_study.train --experiment exp001 --seed 42
"""

import argparse
import csv
import sys

import keras
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.models.cnn_model import build_cnn_model
from src.models.efficientnet_model import build_efficientnet_model
from src.models.mobilenet_model import build_mobilenet_model
from src.seed_study.callbacks import build_callbacks
from src.seed_study.datasets import prepare_dataset
from src.seed_study.environment import environment_record
from src.seed_study.metrics import (
    dump_json,
    first_strict_maximum,
    report_text,
    save_confusion_csv,
    save_confusion_matrix,
    validation_metrics,
)
from src.seed_study.protocol import (
    EXPERIMENTS,
    LOSS,
    METHODOLOGICAL_NOTES,
    METRICS,
    TRAINING_SEEDS,
    UNFREEZE_LAST_N,
    VALIDATION_METRIC_NOTE,
    run_directory,
)
from src.seed_study.split import load_frozen_split, verify_files_exist, verify_membership


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def require_fresh(path):
    if path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing run directory: {path}\n"
            "Delete that directory manually if you intend to repeat the run."
        )


def predict_in_order(model, dataset):
    probabilities = []
    labels = []
    for images, batch_labels in dataset:
        probabilities.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(labels, axis=0).astype(int), np.concatenate(probabilities, axis=0)


def save_predictions(path, val_paths, y_true, probabilities):
    from src.seed_study.protocol import CLASS_NAMES

    y_pred = np.argmax(probabilities, axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "path",
                "true_index",
                "predicted_index",
                "true_class",
                "predicted_class",
                *[f"prob_{name}" for name in CLASS_NAMES],
            ]
        )
        for index, (image_path, truth, prediction, probs) in enumerate(
            zip(val_paths, y_true, y_pred, probabilities)
        ):
            writer.writerow(
                [
                    index,
                    image_path,
                    int(truth),
                    int(prediction),
                    CLASS_NAMES[int(truth)],
                    CLASS_NAMES[int(prediction)],
                    *[f"{float(value):.8f}" for value in probs],
                ]
            )
    return y_pred


def write_history(path, experiment, histories):
    payload = {
        "experiment": experiment,
        "phases": [json_ready(history) for history in histories],
    }
    if EXPERIMENTS[experiment]["selection"] == "best_checkpoint":
        payload["inferred_best_checkpoint"] = first_strict_maximum(histories)
        payload["selection_rule"] = (
            "ModelCheckpoint monitor=val_accuracy mode=max save_best_only, "
            "same object across both phases; evaluation loads best.keras"
        )
    else:
        payload["selection_rule"] = "weights after epoch 10; no checkpoint selection"
    dump_json(path, payload)
    return payload


def train_one(experiment, training_seed):
    if experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {experiment}. Choose one of {tuple(EXPERIMENTS)}")
    if isinstance(training_seed, bool) or training_seed not in TRAINING_SEEDS:
        raise SystemExit(f"Training seed must be one of {TRAINING_SEEDS}")

    spec = EXPERIMENTS[experiment]
    run_dir = run_directory(experiment, training_seed)
    require_fresh(run_dir)

    # The split loader has no seed argument. Load it before seeding RNGs.
    train_paths, val_paths = load_frozen_split()
    verify_membership(train_paths, val_paths)
    verify_files_exist(train_paths, val_paths)

    keras.utils.set_random_seed(training_seed)

    if experiment == "exp001":
        model = build_cnn_model()
        base_model = None
    elif experiment == "exp002":
        model, base_model = build_mobilenet_model()
    else:
        model, base_model = build_efficientnet_model()

    train_ds = prepare_dataset(
        train_paths, experiment, training=True, training_seed=training_seed
    )
    val_ds = prepare_dataset(
        val_paths, experiment, training=False, training_seed=None
    )

    run_dir.mkdir(parents=True)
    dump_json(
        run_dir / "config.json",
        {
            "experiment": experiment,
            "title": spec["title"],
            "training_seed": training_seed,
            "split_seed": 42,
            "split_membership": "frozen manifest; training seed was not used to split",
            "n_train": len(train_paths),
            "n_validation": len(val_paths),
            "metric_set": "VALIDATION",
            "note": VALIDATION_METRIC_NOTE,
            "protocol": spec,
            "methodological_notes": list(METHODOLOGICAL_NOTES),
            "environment": environment_record(),
        },
    )

    if spec["selection"] == "epoch_10":
        model.compile(optimizer="adam", loss=LOSS, metrics=METRICS)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=spec["phase1_epochs"],
        )
        evaluated_path = run_dir / "model.keras"
        model.save(evaluated_path)
        eval_model = model
        history_payload = write_history(run_dir / "history.json", experiment, [history.history])
    else:
        evaluated_path = run_dir / "best.keras"
        callbacks = build_callbacks(evaluated_path)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=LOSS, metrics=METRICS)
        history_phase1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=spec["phase1_epochs"],
            callbacks=callbacks,
        )
        base_model.trainable = True
        for layer in base_model.layers[:-UNFREEZE_LAST_N]:
            layer.trainable = False
        model.compile(optimizer=Adam(learning_rate=1e-5), loss=LOSS, metrics=METRICS)
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=spec["phase2_epochs"],
            callbacks=callbacks,
        )
        # The in-memory model may hold Phase 2 restored weights. The original
        # scripts score the checkpoint file instead.
        eval_model = load_model(evaluated_path)
        history_payload = write_history(
            run_dir / "history.json",
            experiment,
            [history_phase1.history, history_phase2.history],
        )

    y_true, probabilities = predict_in_order(eval_model, val_ds)
    if len(y_true) != len(val_paths):
        raise RuntimeError(
            f"Prediction count {len(y_true)} does not match the frozen "
            f"validation list ({len(val_paths)})"
        )
    y_pred = save_predictions(run_dir / "val_predictions.csv", val_paths, y_true, probabilities)
    metrics = validation_metrics(y_true, y_pred)
    metrics.update(
        {
            "experiment": experiment,
            "training_seed": training_seed,
            "selection_rule": history_payload["selection_rule"],
            "evaluated_weights": evaluated_path.name,
            "inferred_best_checkpoint": history_payload.get("inferred_best_checkpoint"),
        }
    )
    dump_json(run_dir / "metrics.json", metrics)
    (run_dir / "classification_report.txt").write_text(
        report_text(
            y_true,
            y_pred,
            [
                "VALIDATION METRICS — not an independent TrashNet test set",
                f"{spec['title']}",
                f"Training seed: {training_seed}",
                "Split: frozen original seed=42 membership (2022 train / 505 validation)",
                f"Selection: {history_payload['selection_rule']}",
                "Historical unseeded runs are not part of this result.",
            ],
        ),
        encoding="utf-8",
    )
    save_confusion_csv(metrics["confusion_matrix"], run_dir / "confusion_matrix.csv")
    save_confusion_matrix(metrics["confusion_matrix"], run_dir / "confusion_matrix.png")
    dump_json(
        run_dir / "COMPLETE.json",
        {
            "experiment": experiment,
            "training_seed": training_seed,
            "metric_set": "VALIDATION",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "evaluated_weights": evaluated_path.name,
        },
    )
    print(f"VALIDATION accuracy: {metrics['accuracy']:.4f}")
    print(f"VALIDATION macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved run: {run_dir}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Train one seed of one experiment. Does not start the other eight runs."
    )
    parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    parser.add_argument("--seed", required=True, type=int, choices=TRAINING_SEEDS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    train_one(args.experiment, args.seed)


if __name__ == "__main__":
    main()
