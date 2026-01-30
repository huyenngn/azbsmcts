# scripts/train/plot.py
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt

from scripts.common.io import read_jsonl
from utils import utils


def plot_run(
    metrics_path: pathlib.Path, out_dir: pathlib.Path
) -> pathlib.Path | None:
    rows = read_jsonl(metrics_path)
    if not rows:
        return None

    epochs = [r["epoch"] for r in rows]
    loss = [r["loss"] for r in rows]

    run_name = metrics_path.parent.name
    out_path = out_dir / f"{run_name}_loss.png"

    plt.figure()
    plt.plot(epochs, loss, label="loss")

    if all("policy_loss" in r for r in rows):
        plt.plot(
            epochs,
            [float(r["policy_loss"]) for r in rows],
            label="policy_loss",
        )
    if all("value_loss" in r for r in rows):
        plt.plot(
            epochs, [float(r["value_loss"]) for r in rows], label="value_loss"
        )

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=str, default="runs")
    p.add_argument("--out", type=str, default="plots/train")
    p.add_argument("--pattern", type=str, default="*/train_metrics.jsonl")
    p.add_argument("--combined", action="store_true")
    args = p.parse_args()

    runs_dir = pathlib.Path(args.runs)
    out_dir = pathlib.Path(args.out)
    utils.ensure_dir(out_dir)

    metric_files = sorted(runs_dir.glob(args.pattern))
    if not metric_files:
        print(f"No metrics found under {runs_dir}/{args.pattern}")
        return

    written = []
    for mp in metric_files:
        pth = plot_run(mp, out_dir)
        if pth is not None:
            written.append(pth)

    print(f"Wrote {len(written)} plot(s) to {out_dir}")

    if args.combined:
        plt.figure()
        for mp in metric_files:
            rows = read_jsonl(mp)
            if not rows:
                continue
            plt.plot(
                [r["epoch"] for r in rows],
                [r["loss"] for r in rows],
                label=mp.parent.name,
            )
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training loss across runs")
        plt.legend(fontsize="small")
        plt.tight_layout()
        combined_path = out_dir / "combined_loss.png"
        plt.savefig(combined_path)
        plt.close()
        print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
