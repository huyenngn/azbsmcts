# scripts/eval/plot.py
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from scripts.common.io import read_jsonl
from utils import utils


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="inp", type=str, default="plots/eval/eval_sweep.jsonl"
    )
    p.add_argument("--outdir", type=str, default="plots/eval")
    p.add_argument("--x", type=str, default="x_games")
    args = p.parse_args()

    inp = pathlib.Path(args.inp)
    rows = read_jsonl(inp)
    if not rows:
        print(f"No rows in {inp}")
        return

    outdir = pathlib.Path(args.outdir)
    utils.ensure_dir(outdir)

    xs = np.array([r.get(args.x, float("nan")) for r in rows], dtype=float)
    wr_b = np.array(
        [r.get("wr_vs_bsmcts", float("nan")) for r in rows], dtype=float
    )
    wr_r = np.array(
        [r.get("wr_vs_random", float("nan")) for r in rows], dtype=float
    )

    order = np.argsort(xs)
    xs, wr_b, wr_r = xs[order], wr_b[order], wr_r[order]

    plt.figure()
    plt.plot(xs, wr_b, label="AZ vs BS-MCTS")
    plt.plot(xs, wr_r, label="AZ vs Random")
    plt.xlabel(args.x)
    plt.ylabel("win rate")
    plt.ylim(0.0, 1.0)
    plt.title("Evaluation win rate vs training budget")
    plt.legend()
    plt.tight_layout()

    out_path = outdir / "eval_winrate_vs_training.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
