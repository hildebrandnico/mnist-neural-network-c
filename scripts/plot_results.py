#!/usr/bin/env python3
"""Create the figures in docs/images from the CSV files in results/.

Usage:  python3 scripts/plot_results.py
Requires: pandas, matplotlib  (pip install -r requirements.txt)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMAGES = ROOT / "docs" / "images"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_seeds(prefix: str) -> pd.DataFrame:
    """Concatenate the runs of all seeds for one experiment."""
    frames = []
    for i, path in enumerate(sorted(RESULTS.glob(f"{prefix}_seed*.csv")), start=1):
        df = pd.read_csv(path)
        df["seed"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def annotated_heatmap(ax, table: pd.DataFrame, fmt: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(table.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    lo, hi = np.nanmin(table.values), np.nanmax(table.values)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            dark = (v - lo) / (hi - lo + 1e-9) < 0.5
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9,
                    color="white" if dark else "black")
    return im


def plot_lr_epochs():
    df = load_seeds("lr_epochs")
    mean = df.groupby(["learning_rate", "epochs"])["macro_f1"].mean().reset_index()
    table = mean.pivot(index="learning_rate", columns="epochs", values="macro_f1")
    table = table.sort_index(ascending=False)
    table.index = [f"{lr:g}" for lr in table.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})

    for lr, grp in mean.groupby("learning_rate"):
        ax1.plot(grp["epochs"], grp["macro_f1"], marker="o", linewidth=2, label=f"{lr:g}")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sorted(df["epochs"].unique()), sorted(df["epochs"].unique()))
    ax1.set_xlabel("Epochs (log scale)")
    ax1.set_ylabel("Macro F1 on test set (%)")
    ax1.set_title("Learning rate vs. training length")
    ax1.legend(title="Learning rate", loc="lower right")

    im = annotated_heatmap(ax2, table, ".1f", "viridis", vmin=0, vmax=100)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Macro F1 (%), mean of 3 seeds")
    fig.colorbar(im, ax=ax2, label="Macro F1 (%)")

    fig.suptitle("784-64-64-10 network, 10,000 training samples", fontsize=10, color="gray", y=0.02)
    fig.tight_layout()
    fig.savefig(IMAGES / "lr_vs_epochs.png", bbox_inches="tight")
    plt.close(fig)


def plot_train_size():
    df = load_seeds("train_size")
    stats = df.groupby("train_samples").agg(
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        time_mean=("sec_per_epoch", "mean")).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = stats["train_samples"]
    ax1.plot(x, stats["f1_mean"], marker="o", linewidth=2, color="tab:blue", label="Macro F1 (mean ± std, 3 seeds)")
    ax1.fill_between(x, stats["f1_mean"] - stats["f1_std"], stats["f1_mean"] + stats["f1_std"],
                     color="tab:blue", alpha=0.2)
    ax1.set_xscale("log", base=2)
    ticks = [20, 80, 320, 1280, 5120, 20480, 60000]
    ax1.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax1.minorticks_off()
    ax1.set_xlabel("Number of training samples (log scale)")
    ax1.set_ylabel("Macro F1 on test set (%)", color="tab:blue")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(x, stats["time_mean"], marker="s", linestyle="--", color="tab:red", label="Training time per epoch")
    ax2.set_ylabel("Seconds per epoch", color="tab:red")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)

    for xi, yi in zip(x, stats["f1_mean"]):
        if xi >= 320:
            ax1.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color="tab:blue")

    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="center left")
    ax1.set_title("Learning curve: accuracy and cost vs. dataset size  (lr = 0.001, 500 epochs)", pad=14)
    fig.tight_layout()
    fig.savefig(IMAGES / "train_size.png", bbox_inches="tight")
    plt.close(fig)


def plot_hidden_layers():
    df = pd.read_csv(RESULTS / "hidden_layer_sizes.csv")
    f1 = df.pivot(index="hidden1", columns="hidden2", values="macro_f1")
    time = df.pivot(index="hidden1", columns="hidden2", values="sec_per_epoch")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    im1 = annotated_heatmap(ax1, f1, ".1f", "viridis")
    ax1.set_title("Macro F1 on test set (%)")
    ax1.set_xlabel("Neurons in hidden layer 2")
    ax1.set_ylabel("Neurons in hidden layer 1")
    fig.colorbar(im1, ax=ax1)

    im2 = annotated_heatmap(ax2, time, ".2f", "magma")
    ax2.set_title("Training time per epoch (s)")
    ax2.set_xlabel("Neurons in hidden layer 2")
    ax2.set_ylabel("Neurons in hidden layer 1")
    fig.colorbar(im2, ax=ax2)

    fig.suptitle("10,000 training samples, 500 epochs, lr = 0.001", fontsize=10, color="gray", y=0.02)
    fig.tight_layout()
    fig.savefig(IMAGES / "hidden_layers.png", bbox_inches="tight")
    plt.close(fig)


def plot_loss():
    df = pd.read_csv(RESULTS / "default_run_loss.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["epoch"], df["train_loss"], linewidth=2, label="Training loss")
    ax.plot(df["epoch"], df["test_loss"], linewidth=2, linestyle="--",
            label="Test loss (measured before each epoch)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Squared error per sample (log scale)")
    ax.set_title("Training progress (784-64-64-10, lr = 0.1, 60,000 samples)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "loss_curve.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    plot_lr_epochs()
    plot_train_size()
    plot_hidden_layers()
    plot_loss()
    print(f"Figures written to {IMAGES}")
