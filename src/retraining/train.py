"""Corrected EXP-003 and EXP-004 training entry point.

Full training is refused unless approve_training=True. The preflight does not
pass that flag. Evidence files do not change the fit arguments or the
checkpoint rule.
"""

import json
import shutil

import numpy as np
from tensorflow.keras.models import load_model

from src.corrected_baselines.evidence import (
    HistoryStore,
    build_evaluation,
    capture_warnings,
    write_json,
    write_predictions,
    write_run_header,
    write_status,
)
from src.retraining.data import (
    load_batch,
    load_split,
    steps_per_epoch,
    training_dataset,
    validation_dataset,
)
from src.retraining.model import (
    EpochLog,
    build_model,
    compile_phase1,
    compile_phase2,
    make_callbacks,
)
from src.retraining.protocol import (
    BATCH_SIZE,
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    PROTECTED_PREFIXES,
    PROJECT_ROOT,
    run_directory,
)


def _guard_output(run_dir):
    relative = run_dir.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    for prefix in PROTECTED_PREFIXES:
        if relative.startswith(prefix):
            raise RuntimeError(f"Refusing to write into protected path {relative}")
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")


def _best_row(records, phase):
    rows = [row for row in records if row["phase"] == phase]
    if not rows:
        raise RuntimeError(f"Phase {phase} produced no epoch records")
    chosen = rows[0]
    for row in rows[1:]:
        if row["val_accuracy"] > chosen["val_accuracy"]:
            chosen = row
    return chosen


def select_published(run_dir, records):
    phase1 = _best_row(records, 1)
    phase2 = _best_row(records, 2)
    if phase2["val_accuracy"] > phase1["val_accuracy"]:
        chosen = phase2
        source_name = "phase2_best.keras"
    else:
        chosen = phase1
        source_name = "phase1_best.keras"
    source = run_dir / source_name
    published = run_dir / "published.keras"
    shutil.copyfile(source, published)
    selection = {
        "file": source_name,
        "published_file": "published.keras",
        "phase": chosen["phase"],
        "epoch_in_phase": chosen["epoch_in_phase"],
        "global_epoch": chosen["global_epoch"],
        "val_accuracy": chosen["val_accuracy"],
        "tie_rule": "equal val_accuracy keeps phase 1",
    }
    write_json(run_dir / "selection.json", selection)
    return selection


def _predict_validation(model, paths, labels):
    probabilities = []
    observed = []
    for start in range(0, len(paths), BATCH_SIZE):
        images, batch_labels = load_batch(
            paths[start:start + BATCH_SIZE],
            labels[start:start + BATCH_SIZE],
        )
        probabilities.append(model(images, training=False).numpy())
        observed.extend(int(label) for label in batch_labels)
    if observed != list(labels):
        raise RuntimeError("Validation predictions left manifest order")
    return np.concatenate(probabilities, axis=0)


def _write_evaluation(run_dir, experiment, training_seed, model, paths, labels, selection):
    probabilities = _predict_validation(model, paths, labels)
    write_predictions(run_dir / "val_predictions.csv", paths, labels, probabilities)
    val_loss, val_accuracy = model.evaluate(
        validation_dataset(paths, labels),
        steps=steps_per_epoch(len(paths)),
        verbose=0,
    )
    evaluation = build_evaluation(
        experiment,
        training_seed,
        labels,
        probabilities,
        val_loss,
        selection,
    )
    if abs(evaluation["accuracy"] - float(val_accuracy)) > 1e-6:
        raise RuntimeError(
            f"Prediction accuracy {evaluation['accuracy']} disagrees with "
            f"evaluate() accuracy {val_accuracy}"
        )
    write_json(run_dir / "evaluation.json", evaluation)
    return evaluation


def run(experiment, training_seed, approve_training=False):
    if not approve_training:
        raise RuntimeError(
            "Full training is not approved. Call run(..., approve_training=True) "
            "only after the preflight has been accepted."
        )
    run_dir = run_directory(experiment, training_seed)
    _guard_output(run_dir)
    train_paths, train_labels, val_paths, val_labels = load_split()
    train_ds = training_dataset(train_paths, train_labels, training_seed)
    val_ds = validation_dataset(val_paths, val_labels)
    train_steps = steps_per_epoch(len(train_paths))
    val_steps = steps_per_epoch(len(val_paths))
    model, base = build_model(experiment, training_seed)
    records = []

    run_dir.mkdir(parents=True)
    try:
        with capture_warnings(run_dir / "warnings.txt"):
            write_run_header(run_dir, experiment, training_seed, train_paths, val_paths)
            store = HistoryStore(run_dir / "history.json", experiment, training_seed)
            store.start_phase(1)
            compile_phase1(model, base)
            checkpoint, early_stopping = make_callbacks(run_dir, 1)
            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=PHASE1_EPOCHS,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                callbacks=[
                    checkpoint,
                    early_stopping,
                    EpochLog(1, 0, records, on_record=store.adopt),
                ],
                verbose=2,
            )
            phase1_epochs_ran = len(history1.history["loss"])
            store.start_phase(2)
            compile_phase2(model, base)
            checkpoint, early_stopping = make_callbacks(run_dir, 2)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=PHASE2_EPOCHS,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                callbacks=[
                    checkpoint,
                    early_stopping,
                    EpochLog(2, phase1_epochs_ran, records, on_record=store.adopt),
                ],
                verbose=2,
            )
            selection = select_published(run_dir, records)
            published = load_model(run_dir / "published.keras")
            evaluation = _write_evaluation(
                run_dir,
                experiment,
                training_seed,
                published,
                val_paths,
                val_labels,
                selection,
            )
            write_status(run_dir / "status.json", "completed")
            return evaluation
    except Exception as exc:
        if run_dir.exists():
            try:
                write_status(run_dir / "status.json", "failed", exception=exc)
            except Exception:
                pass
        if run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise


if __name__ == "__main__":
    raise SystemExit(
        "Full training is not started from this module. "
        "Use src.retraining.preflight for the smoke test."
    )
