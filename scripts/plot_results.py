#!/usr/bin/env python3
"""Plot vLLM serving benchmark results."""

import argparse
import csv
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_throughput_vs_concurrency(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[as_int(row, "input_tokens")].append(row)

    plt.figure(figsize=(8, 5))
    for input_tokens in sorted(grouped):
        group = sorted(grouped[input_tokens], key=lambda r: as_int(r, "concurrency"))
        x = [as_int(r, "concurrency") for r in group]
        y = [as_float(r, "throughput_tok_s") for r in group]
        plt.plot(x, y, marker="o", label=f"input={input_tokens}")

    plt.title("vLLM Throughput vs Concurrency")
    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (output tokens/sec)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Context length")
    plt.tight_layout()
    plt.savefig(output_dir / "throughput_vs_concurrency.png", dpi=160)
    plt.close()


def plot_latency_vs_context(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[as_int(row, "concurrency")].append(row)

    plt.figure(figsize=(8, 5))
    for concurrency in sorted(grouped):
        group = sorted(grouped[concurrency], key=lambda r: as_int(r, "input_tokens"))
        x = [as_int(r, "input_tokens") for r in group]
        y = [as_float(r, "p95_latency_s") for r in group]
        plt.plot(x, y, marker="o", label=f"concurrency={concurrency}")

    plt.title("vLLM P95 Latency vs Context Length")
    plt.xlabel("Input context length (approx. tokens)")
    plt.ylabel("P95 latency (sec)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Concurrency")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_vs_context.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/serving_benchmark.csv")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    plot_throughput_vs_concurrency(rows, output_dir)
    plot_latency_vs_context(rows, output_dir)

    print(f"Wrote {output_dir / 'throughput_vs_concurrency.png'}")
    print(f"Wrote {output_dir / 'latency_vs_context.png'}")


if __name__ == "__main__":
    main()
