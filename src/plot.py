import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_run(metrics_path: Path, out_dir: Path):
    rows = read_jsonl(metrics_path)
    if not rows:
        return

    epochs = [r["epoch"] for r in rows]
    loss = [r["loss"] for r in rows]
    has_ploss = all("policy_loss" in r for r in rows)
    has_vloss = all("value_loss" in r for r in rows)

    run_name = metrics_path.parent.name
    out_path = out_dir / f"{run_name}_loss.png"

    plt.figure()
    plt.plot(epochs, loss, label="loss")

    if has_ploss:
        ploss = [float(r["policy_loss"]) for r in rows]
        plt.plot(epochs, ploss, label="policy_loss")
    if has_vloss:
        vloss = [float(r["value_loss"]) for r in rows]
        plt.plot(epochs, vloss, label="value_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=str, default="runs", help="runs directory"
    )
    parser.add_argument(
        "--out", type=str, default="plots", help="output directory"
    )
    parser.add_argument("--pattern", type=str, default="*/train_metrics.jsonl")
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(runs_dir.glob(args.pattern))
    if not metric_files:
        print(f"No metrics found under {runs_dir}/{args.pattern}")
        return

    written = []
    for mp in metric_files:
        p = plot_run(mp, out_dir)
        if p is not None:
            written.append(p)

    print(f"Wrote {len(written)} plot(s) to {out_dir}")

    if args.combined:
        plt.figure()
        for mp in metric_files:
            rows = read_jsonl(mp)
            if not rows:
                continue
            epochs = [r["epoch"] for r in rows]
            loss = [r["loss"] for r in rows]
            plt.plot(epochs, loss, label=mp.parent.name)
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
