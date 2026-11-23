#!/usr/bin/env python3
"""Run the hyperparameter sweeps from the README with the mnist-nn binary.

Each preset reproduces one experiment. The binary appends one row per run
(hyperparameters + test metrics) to <out>/experiments.csv; once a preset has
finished, that file is copied to results/<preset>.csv, where
scripts/plot_results.py picks it up.

Usage:
    make && python3 scripts/run_experiments.py lr_epochs
    python3 scripts/run_experiments.py --list
    python3 scripts/run_experiments.py default_loss   # loss curve of the default model

Note: the full presets take hours (e.g. 1280 epochs or 60,000 samples x 500
epochs). Use --dry-run to see what would be run.
"""
import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "mnist-nn"
RESULTS = ROOT / "results"

SIZES = [16, 32, 64, 128, 256]

PRESETS = {
    # Figure 1: learning rate vs. number of epochs (5 x 9 x 3 = 135 runs)
    "lr_epochs": {
        "hidden": ["64,64"],
        "train_n": [10000],
        "epochs": [5, 10, 20, 40, 80, 160, 320, 640, 1280],
        "lr": [1.0, 0.1, 0.01, 0.001, 0.0001],
        "seed": [1, 2, 3],
    },
    # Figure 2: learning curve over the training-set size (13 x 3 = 39 runs)
    "train_size": {
        "hidden": ["64,64"],
        "train_n": [20 * 2**i for i in range(12)] + [60000],
        "epochs": [500],
        "lr": [0.001],
        "seed": [1, 2, 3],
    },
    # Figure 3: width of the two hidden layers (5 x 5 = 25 runs)
    "hidden_layer_sizes": {
        "hidden": [f"{a},{b}" for a in SIZES for b in SIZES],
        "train_n": [10000],
        "epochs": [500],
        "lr": [0.001],
        "seed": [1],
    },
}
TEST_N = 3000


def run(cmd: list[str], dry_run: bool) -> None:
    print("  " + " ".join(cmd[1:]))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  failed (exit code {result.returncode}): {result.stderr.strip()}")


def run_preset(name: str, dry_run: bool) -> None:
    grid = PRESETS[name]
    out_dir = ROOT / "output" / name
    if not dry_run:
        # Start from a clean file so results/<name>.csv only holds this sweep.
        (out_dir / "experiments.csv").unlink(missing_ok=True)

    keys = list(grid)
    runs = list(itertools.product(*grid.values()))
    print(f"{name}: {len(runs)} runs")
    for i, values in enumerate(runs, start=1):
        p = dict(zip(keys, values))
        print(f"[{i}/{len(runs)}]", end="")
        run([str(BINARY),
             "--hidden", p["hidden"],
             "--train-n", str(p["train_n"]),
             "--test-n", str(TEST_N),
             "--epochs", str(p["epochs"]),
             "--lr", str(p["lr"]),
             "--seed", str(p["seed"]),
             "--out", str(out_dir)], dry_run)

    if not dry_run:
        RESULTS.mkdir(exist_ok=True)
        shutil.copy(out_dir / "experiments.csv", RESULTS / f"{name}.csv")
        print(f"Results written to {RESULTS / f'{name}.csv'}")


def run_default_loss(dry_run: bool) -> None:
    """Default model on the full dataset; its per-epoch loss is figure 4."""
    out_dir = ROOT / "output" / "default_loss"
    print("default_loss: 1 run")
    run([str(BINARY), "--seed", "42", "--out", str(out_dir)], dry_run)
    if not dry_run:
        RESULTS.mkdir(exist_ok=True)
        shutil.copy(out_dir / "loss.csv", RESULTS / "default_run_loss.csv")
        print(f"Results written to {RESULTS / 'default_run_loss.csv'}")


def main() -> None:
    choices = [*PRESETS, "default_loss"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("presets", nargs="*", choices=choices, metavar="PRESET",
                        help=f"one or more of: {', '.join(choices)}")
    parser.add_argument("--list", action="store_true", help="list the presets and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    if args.list or not args.presets:
        for name, grid in PRESETS.items():
            n = len(list(itertools.product(*grid.values())))
            print(f"{name:20s} {n:4d} runs")
        print(f"{'default_loss':20s} {1:4d} run")
        return

    if not BINARY.exists() and not args.dry_run:
        sys.exit("error: build the program first with `make`")

    for name in args.presets:
        if name == "default_loss":
            run_default_loss(args.dry_run)
        else:
            run_preset(name, args.dry_run)


if __name__ == "__main__":
    main()
