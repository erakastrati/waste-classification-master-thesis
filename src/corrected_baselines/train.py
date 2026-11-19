"""Train one corrected EXP-001 or EXP-002 seed.

Full training is refused unless approve_training=True. Running this file
does not pass that flag.
"""

import inspect

import numpy as np
from tensorflow.keras.models import load_model

from src.corrected_baselines.data import (
    PREPROCESS,
    load_batch,
    load_split,
    steps_per_epoch,
    training_dataset,
    validation_dataset,
)
from src.corrected_baselines.evidence import (
    HistoryStore,
    build_evaluation,
    capture_warnings,
    history_callback,
    write_json,
    write_predictions,
    write_run_header,
    write_status,
)
from src.corrected_baselines.models import (
    build_exp001,
    build_exp002,
    compile_exp001,
    compile_exp002_phase1,
    compile_exp002_phase2,
    exp002_callback_list,
)
from src.corrected_baselines.protocol import (
    BASELINE_EXPERIMENTS,
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    SELECTION_RULE,
    TRAINING_SEEDS,
    assert_safe_output,
    run_directory,
)


def _guard(experiment, training_seed, approve_training):
    if approve_training is not True:
        raise RuntimeError(
            "Full training is not approved. Call run(..., approve_training=True) "
            "only after this preflight has been accepted."
        )
    if experiment not in BASELINE_EXPERIMENTS:
        raise RuntimeError(f"{experiment} is not trained by this package")
    if training_seed not in TRAINING_SEEDS:
        raise RuntimeError(f"Training seed must be one of {TRAINING_SEEDS}")
    run_dir = assert_safe_output(run_directory(experiment, training_seed))
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")
    return run_dir


def _predict_validation(model, paths, labels, preprocess):
    probabilities = []
    observed = []
    for start in range(0, len(paths), 32):
        images, batch_labels = load_batch(
            paths[start:start + 32],
            labels[start:start + 32],
            preprocess,
        )
        probabilities.append(model(images, training=False).numpy())
        observed.extend(int(label) for label in batch_labels)
    if observed != list(labels):
        raise RuntimeError("Validation predictions left manifest order")
    return np.concatenate(probabilities, axis=0)


def _evaluation(run_dir, experiment, training_seed, model, paths, labels, selection):
    preprocess = PREPROCESS[experiment]
    probabilities = _predict_validation(model, paths, labels, preprocess)
    write_predictions(run_dir / "val_predictions.csv", paths, labels, probabilities)
    loss, accuracy = model.evaluate(
        validation_dataset(paths, labels, preprocess),
        steps=steps_per_epoch(len(paths)),
        verbose=0,
    )
    evaluation = build_evaluation(
        experiment, training_seed, labels, probabilities, loss, selection
    )
    if abs(evaluation["accuracy"] - float(accuracy)) > 1e-6:
        raise RuntimeError(
            f"Prediction accuracy {evaluation['accuracy']} disagrees with "
            f"evaluate() accuracy {accuracy}"
        )
    write_json(run_dir / "evaluation.json", evaluation)
    return evaluation


def _run_exp001(run_dir, training_seed, train_paths, train_labels, val_paths, val_labels, store):
    model = build_exp001(training_seed)
    compile_exp001(model)
    preprocess = PREPROCESS["EXP-001"]
    store.start_phase(1)
    history = model.fit(
        training_dataset(train_paths, train_labels, training_seed, preprocess),
        validation_data=validation_dataset(val_paths, val_labels, preprocess),
        epochs=PHASE1_EPOCHS,
        steps_per_epoch=steps_per_epoch(len(train_paths)),
        validation_steps=steps_per_epoch(len(val_paths)),
        callbacks=[history_callback(store, 1, 0)],
        verbose=2,
    )
    if len(history.history["loss"]) != PHASE1_EPOCHS:
        raise RuntimeError("EXP-001 did not complete 10 epochs")
    model_path = run_dir / "model.keras"
    model.save(model_path)
    evaluated = load_model(model_path)
    selection = {
        "file": "model.keras",
        "phase": 1,
        "epoch": PHASE1_EPOCHS,
        "epoch_in_phase": PHASE1_EPOCHS,
        "global_epoch": PHASE1_EPOCHS,
        "rule": SELECTION_RULE["EXP-001"],
    }
    write_json(run_dir / "selection.json", selection)
    return _evaluation(
        run_dir, "EXP-001", training_seed, evaluated, val_paths, val_labels, selection
    )


def _run_exp002(run_dir, training_seed, train_paths, train_labels, val_paths, val_labels, store):
    model, base = build_exp002(training_seed)
    preprocess = PREPROCESS["EXP-002"]
    callbacks = exp002_callback_list(run_dir / "best.keras")
    compile_exp002_phase1(model)
    store.start_phase(1)
    history1 = model.fit(
        training_dataset(train_paths, train_labels, training_seed, preprocess),
        validation_data=validation_dataset(val_paths, val_labels, preprocess),
        epochs=PHASE1_EPOCHS,
        steps_per_epoch=steps_per_epoch(len(train_paths)),
        validation_steps=steps_per_epoch(len(val_paths)),
        callbacks=callbacks + [history_callback(store, 1, 0)],
        verbose=2,
    )
    compile_exp002_phase2(model, base)
    store.start_phase(2)
    history2 = model.fit(
        training_dataset(train_paths, train_labels, training_seed, preprocess),
        validation_data=validation_dataset(val_paths, val_labels, preprocess),
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=steps_per_epoch(len(train_paths)),
        validation_steps=steps_per_epoch(len(val_paths)),
        callbacks=callbacks + [history_callback(store, 2, len(store.records))],
        verbose=2,
    )
    best_path = run_dir / "best.keras"
    if not best_path.is_file():
        raise RuntimeError("EXP-002 did not write best.keras")
    evaluated = load_model(best_path)
    selection = _first_strict_maximum(history1.history, history2.history)
    selection["file"] = "best.keras"
    selection["rule"] = SELECTION_RULE["EXP-002"]
    write_json(run_dir / "selection.json", selection)
    return _evaluation(
        run_dir, "EXP-002", training_seed, evaluated, val_paths, val_labels, selection
    )


def _first_strict_maximum(phase1, phase2):
    best_value = float("-inf")
    best = None
    global_epoch = 0
    for phase, history in ((1, phase1), (2, phase2)):
        for epoch, value in enumerate(history["val_accuracy"], start=1):
            global_epoch += 1
            value = float(value)
            if value > best_value:
                best_value = value
                best = {
                    "phase": phase,
                    "epoch_in_phase": epoch,
                    "global_epoch": global_epoch,
                    "val_accuracy": value,
                }
    return best


def run(experiment, training_seed, approve_training=False):
    run_dir = _guard(experiment, training_seed, approve_training)
    train_paths, train_labels, val_paths, val_labels = load_split()
    run_dir.mkdir(parents=True)
    try:
        with capture_warnings(run_dir / "warnings.txt"):
            write_run_header(run_dir, experiment, training_seed, train_paths, val_paths)
            store = HistoryStore(run_dir / "history.json", experiment, training_seed)
            if experiment == "EXP-001":
                evaluation = _run_exp001(
                    run_dir, training_seed, train_paths, train_labels, val_paths, val_labels, store
                )
            else:
                evaluation = _run_exp002(
                    run_dir, training_seed, train_paths, train_labels, val_paths, val_labels, store
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


def trainer_uses_shared_exp002_callbacks():
    """True when both EXP-002 fit calls receive the one callback list."""
    source = inspect.getsource(_run_exp002)
    return source.count("exp002_callback_list(") == 1 and source.count("callbacks=callbacks") == 2


if __name__ == "__main__":
    raise SystemExit(
        "Full training is not started from this module. "
        "Use python -m src.corrected_baselines.preflight for the smoke test."
    )
