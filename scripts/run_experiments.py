#!/usr/bin/env python3
"""Run a hyperparameter grid search with the mnist-nn binary.

Every run appends one row (hyperparameters + test metrics) to
output/experiments.csv. Edit the GRID below to choose what to sweep.

Usage:  make && python3 scripts/run_experiments.py
"""
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "mnist-nn"

GRID = {
    "hidden": ["64,64"],              # e.g. ["16,16", "64,64", "256,256"]
    "train_n": [10000],               # e.g. [1000, 10000, 60000]
    "epochs": [5, 10, 20, 40],
    "lr": [1.0, 0.1, 0.01, 0.001],
    "seed": [1, 2, 3],                # repeat each setting with different initialisations
}
TEST_N = 3000


def main() -> None:
    if not BINARY.exists():
        sys.exit("error: build the program first with `make`")

    keys = list(GRID)
    runs = list(itertools.product(*GRID.values()))
    print(f"Running {len(runs)} experiments...")

    for i, values in enumerate(runs, start=1):
        p = dict(zip(keys, values))
        print(f"[{i}/{len(runs)}] {p}")
        cmd = [
            str(BINARY),
            "--hidden", p["hidden"],
            "--train-n", str(p["train_n"]),
            "--test-n", str(TEST_N),
            "--epochs", str(p["epochs"]),
            "--lr", str(p["lr"]),
            "--seed", str(p["seed"]),
        ]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  failed (exit code {result.returncode}): {result.stderr.strip()}")

    print("Done. Results were appended to output/experiments.csv")


if __name__ == "__main__":
    main()
