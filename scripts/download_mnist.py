#!/usr/bin/env python3
"""Download MNIST and convert it to the CSV format used by mnist-nn.

Output (in ./data):
    mnist_train.csv   60,000 rows
    mnist_test.csv    10,000 rows

Each row holds 784 pixel values (0-255) followed by the label:
    x1,x2,...,x784,y

Only the Python standard library is required.
"""
import gzip
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.URLError:
        # Some Python installs (e.g. python.org builds on macOS) ship without
        # CA certificates; fall back to the system's curl in that case.
        return subprocess.run(["curl", "-fsSL", url], check=True, capture_output=True).stdout


def fetch(name: str) -> bytes:
    for mirror in MIRRORS:
        url = mirror + name
        try:
            print(f"  downloading {url}")
            return gzip.decompress(download(url))
        except Exception as exc:  # try the next mirror
            print(f"    failed: {exc}")
    sys.exit(f"error: could not download {name} from any mirror")


def parse_images(raw: bytes) -> list[bytes]:
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, "not an IDX image file"
    size = rows * cols
    return [raw[16 + i * size: 16 + (i + 1) * size] for i in range(n)]


def parse_labels(raw: bytes) -> bytes:
    magic, n = struct.unpack(">II", raw[:8])
    assert magic == 2049, "not an IDX label file"
    return raw[8: 8 + n]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for split, (img_file, lbl_file) in FILES.items():
        out = DATA_DIR / f"mnist_{split}.csv"
        if out.exists():
            print(f"{out.name} already exists, skipping")
            continue

        print(f"Preparing {out.name}")
        images = parse_images(fetch(img_file))
        labels = parse_labels(fetch(lbl_file))
        assert len(images) == len(labels)

        header = ",".join(f"x{i + 1}" for i in range(784)) + ",y\n"
        with out.open("w") as f:
            f.write(header)
            for img, label in zip(images, labels):
                f.write(",".join(map(str, img)) + f",{label}\n")
        print(f"  wrote {len(images)} samples to {out}")


if __name__ == "__main__":
    main()
