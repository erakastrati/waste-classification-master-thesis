"""Callback objects copied from the original EXP-002 and EXP-003 scripts."""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.seed_study.protocol import CHECKPOINT_KWARGS, EARLY_STOPPING_KWARGS, assert_safe_output


def build_callbacks(checkpoint_path):
    """Return one checkpoint and one early-stopping object.

    The trainer passes this same list to Phase 1 and Phase 2. EarlyStopping
    resets its own best score at the start of each fit call. ModelCheckpoint
    keeps its best score, so the file remains the best val_accuracy across
    both phases. An equal score does not replace the file (numpy.greater).
    """
    path = assert_safe_output(checkpoint_path)
    if path.name != "best.keras":
        raise ValueError(f"Checkpoint filename must be best.keras, received {path.name}")
    checkpoint = ModelCheckpoint(filepath=str(path), **CHECKPOINT_KWARGS)
    early_stop = EarlyStopping(**EARLY_STOPPING_KWARGS)
    return [checkpoint, early_stop]
