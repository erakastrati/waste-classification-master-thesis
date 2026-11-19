# EXP-011 run directory

Reserved for a future Garbage v2 fine-tune. Phase 1 did not audit this experiment and did not create a run.

The historical script `src/training/train_garbage_finetune.py` loads `results/efficientnet_household/efficientnet_household.keras`.

## Unique run folders

Create one new directory per attempt:

`runs/<UTC timestamp>_<short label>/`

If the chosen directory already exists, stop. Do not delete, rename, or replace files inside an existing run directory. Do not write checkpoints, logs, or reports into this directory itself.
