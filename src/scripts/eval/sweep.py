# scripts/eval/sweep.py
from __future__ import annotations

import argparse
import pathlib
from datetime import datetime

import pyspiel

from scripts.common.config import GameConfig, SamplerConfig, SearchConfig
from scripts.common.io import read_json, write_jsonl
from scripts.eval.match import run_match


def az_winrate(res: dict, az_label: str = "azbsmcts") -> float:
    items = list(res.items())
    r1 = items[0][1]
    r2 = items[1][1]
    # r1/r2 are Result dataclasses in-process; we only need wins and games
    az_wins = (
        (r1.p0_wins + r2.p1_wins)
        if az_label in items[0][0]
        else (r1.p1_wins + r2.p0_wins)
    )
    total = r1.games + r2.games
    return az_wins / max(1, total)


def find_run_dirs(runs_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        [p for p in runs_root.glob("*") if (p / "model.pt").exists()]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--game", type=str, default="phantom_go")
    p.add_argument("--game-params", type=str, default='{"board_size": 9}')

    p.add_argument("--n", type=int, default=20)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--S", type=int, default=4)

    p.add_argument("--runs", type=str, default="runs")
    p.add_argument("--out", type=str, default="plots/eval/eval_sweep.jsonl")

    p.add_argument("--x-axis", type=str, default="games")  # from run config
    p.add_argument("--num-particles", type=int, default=32)
    p.add_argument("--opp-tries", type=int, default=8)
    p.add_argument("--rebuild-tries", type=int, default=200)

    args = p.parse_args()

    game_cfg = GameConfig.from_cli(args.game, args.game_params)
    search_cfg = SearchConfig(T=args.T, S=args.S)
    sampler_cfg = SamplerConfig(
        args.num_particles, args.opp_tries, args.rebuild_tries
    )

    game = pyspiel.load_game(game_cfg.name, game_cfg.params)

    runs_root = pathlib.Path(args.runs)
    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        print(f"No run dirs with model.pt found under {runs_root}")
        return

    rows = []
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for rd in run_dirs:
        cfg_path = rd / "config.json"
        cfg = read_json(cfg_path) if cfg_path.exists() else {}
        model_path = str(rd / "model.pt")

        # evaluate vs baselines
        res_b = run_match(
            game=game,
            a="azbsmcts",
            b="bsmcts",
            n=args.n,
            search_cfg=search_cfg,
            sampler_cfg=sampler_cfg,
            seed=args.seed,
            device=args.device,
            model_path=model_path,
            run_id=f"sweep_{sweep_id}_{rd.name}",
        )
        wr_b = az_winrate(res_b)

        res_r = run_match(
            game=game,
            a="azbsmcts",
            b="random",
            n=args.n,
            search_cfg=search_cfg,
            sampler_cfg=sampler_cfg,
            seed=args.seed + 999,
            device=args.device,
            model_path=model_path,
            run_id=f"sweep_{sweep_id}_{rd.name}_random",
        )
        wr_r = az_winrate(res_r)

        x = float("nan")
        if args.x_axis == "games":
            # expects your train config to store games at top-level or inside budget; your current train stores args directly :contentReference[oaicite:10]{index=10}
            if isinstance(cfg, dict):
                if "games" in cfg:
                    x = float(cfg["games"])
                elif (
                    "budget" in cfg
                    and isinstance(cfg["budget"], dict)
                    and "games" in cfg["budget"]
                ):
                    x = float(cfg["budget"]["games"])

        row = {
            "run_dir": rd.name,
            "model_path": model_path,
            "x_games": x,
            "wr_vs_bsmcts": float(wr_b),
            "wr_vs_random": float(wr_r),
        }
        rows.append(row)
        print(f"{rd.name}: wr_vs_bsmcts={wr_b:.3f} wr_vs_random={wr_r:.3f}")

    out_path = pathlib.Path(args.out)
    write_jsonl(out_path, rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
