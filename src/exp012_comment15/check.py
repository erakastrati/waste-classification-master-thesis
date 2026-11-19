"""Static and split checks for the EXP-012 package.

Running this module does not build a model and does not call fit().
A completed benchmark directory is reported as preserved. The benchmark
runner still refuses to overwrite it.
"""

import sys

from src.exp012_comment15.safety import (
    ALLOWED_BATCH_SIZES,
    static_safety_report,
    run_directory,
)
from src.seed_study.protocol import EXPECTED_TRAIN_N, EXPECTED_VAL_N
from src.seed_study.split import (
    load_frozen_split,
    verify_files_exist,
    verify_membership,
)


def verify_split():
    train_paths, val_paths = load_frozen_split()
    verify_membership(train_paths, val_paths)
    verify_files_exist(train_paths, val_paths)
    overlap = len(set(train_paths) & set(val_paths))
    if overlap != 0:
        raise RuntimeError(f"train/validation overlap is {overlap}")
    if len(train_paths) != EXPECTED_TRAIN_N or len(val_paths) != EXPECTED_VAL_N:
        raise RuntimeError(
            f"Split counts are {len(train_paths)}/{len(val_paths)}, "
            f"expected {EXPECTED_TRAIN_N}/{EXPECTED_VAL_N}"
        )
    return {
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "overlap": overlap,
    }


def main():
    errors = static_safety_report()
    for message in errors:
        print(f"FAIL static: {message}")
    if errors:
        raise SystemExit(1)
    print("PASS static safety checks")
    split = verify_split()
    print(
        "PASS split "
        f"n_train={split['n_train']} n_val={split['n_val']} overlap={split['overlap']}"
    )
    for batch_size in ALLOWED_BATCH_SIZES:
        path = run_directory(batch_size)
        if path.exists():
            print(f"PASS benchmark directory preserved: {path}")
        else:
            print(f"PASS benchmark directory absent: {path}")
    print("PASS preflight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
