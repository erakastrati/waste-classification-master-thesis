"""Aggregate completed controlled runs into mean ± sample standard deviation.

Does not read historical result directories and does not train.
"""

import sys

from src.seed_study.metrics import aggregate_results


def main():
    summary = aggregate_results()
    for name, block in summary["experiments"].items():
        print(
            f"{name} VALIDATION accuracy {block['accuracy_mean_pm_std']} | "
            f"macro F1 {block['macro_f1_mean_pm_std']}"
        )
    print("Wrote results/seed_study/summary.json")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
