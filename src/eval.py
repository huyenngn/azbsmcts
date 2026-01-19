import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyspiel

from agents import AZBSMCTSAgent, BSMCTSAgent
from belief.samplers.particle import (
    ParticleBeliefSampler,
    ParticleDeterminizationSampler,
)


@dataclass
class Result:
    games: int = 0
    p0_wins: int = 0
    p1_wins: int = 0
    draws: int = 0
    p0_return_sum: float = 0.0
    p1_return_sum: float = 0.0


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _obs_size_for_player(game: pyspiel.Game, player_id: int) -> int:
    s = game.new_initial_state()
    # Strict: always specify player id
    return len(s.observation_tensor(player_id))


def make_agent(
    kind: str,
    player_id: int,
    game: pyspiel.Game,
    T: int,
    S: int,
    seed: int,
    device: str,
):
    num_actions = game.num_distinct_actions()

    if kind == "random":
        return None, None

    particle = ParticleBeliefSampler(
        game=game,
        ai_id=player_id,
        num_particles=32,
        opp_tries_per_particle=8,
        rebuild_max_tries=200,
        seed=seed * 999 + player_id,
    )
    sampler = ParticleDeterminizationSampler(particle)

    if kind == "bsmcts":
        return (
            BSMCTSAgent(
                player_id=player_id,
                num_actions=num_actions,
                sampler=sampler,
                T=T,
                S=S,
                seed=seed,
            ),
            particle,
        )

    if kind == "azbsmcts":
        obs_size = _obs_size_for_player(game, player_id=player_id)
        return (
            AZBSMCTSAgent(
                player_id=player_id,
                num_actions=num_actions,
                obs_size=obs_size,
                sampler=sampler,
                T=T,
                S=S,
                seed=seed,
                device=device,
                model_path=os.environ.get("AZ_MODEL_PATH", None),
            ),
            particle,
        )

    raise ValueError(f"Unknown agent kind: {kind}")


def select_action(
    kind: str, agent, state: pyspiel.State, rng: random.Random
) -> int:
    if kind == "random" or agent is None:
        return rng.choice(state.legal_actions())
    return agent.select_action(state)


def play_game(
    game: pyspiel.Game,
    kind0: str,
    kind1: str,
    T: int,
    S: int,
    seed: int,
    device: str,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    state = game.new_initial_state()

    a0, p0 = make_agent(kind0, 0, game, T, S, seed * 17 + 1, device)
    a1, p1 = make_agent(kind1, 1, game, T, S, seed * 17 + 2, device)

    while not state.is_terminal():
        actor = state.current_player()
        if actor == 0:
            action = select_action(kind0, a0, state, rng)
        else:
            action = select_action(kind1, a1, state, rng)

        state.apply_action(action)

        # update BOTH belief samplers via observation strings (sampler ignores hidden opponent action)
        if p0 is not None:
            p0.step(actor=actor, action=action, real_state_after=state)
        if p1 is not None:
            p1.step(actor=actor, action=action, real_state_after=state)

    r = state.returns()
    return float(r[0]), float(r[1])


def update_result(res: Result, r0: float, r1: float):
    res.games += 1
    res.p0_return_sum += r0
    res.p1_return_sum += r1
    if r0 > r1:
        res.p0_wins += 1
    elif r1 > r0:
        res.p1_wins += 1
    else:
        res.draws += 1


def summarize(label: str, res: Result):
    p0_wr = res.p0_wins / max(1, res.games)
    p1_wr = res.p1_wins / max(1, res.games)
    dr = res.draws / max(1, res.games)
    print(f"\n{label}")
    print(f"games: {res.games}")
    print(
        f"p0 winrate: {p0_wr:.3f} | p1 winrate: {p1_wr:.3f} | draws: {dr:.3f}"
    )
    print(
        f"mean returns: p0={res.p0_return_sum / res.games:.3f}, p1={res.p1_return_sum / res.games:.3f}"
    )


def run_match(
    game: pyspiel.Game,
    a: str,
    b: str,
    n: int,
    T: int,
    S: int,
    seed: int,
    device: str,
) -> Dict[str, Result]:
    out: Dict[str, Result] = {}

    res_ab = Result()
    for i in range(n):
        r0, r1 = play_game(game, a, b, T, S, seed + i, device)
        update_result(res_ab, r0, r1)
    out[f"{a}(p0) vs {b}(p1)"] = res_ab

    res_ba = Result()
    for i in range(n):
        r0, r1 = play_game(game, b, a, T, S, seed + 10_000 + i, device)
        update_result(res_ba, r0, r1)
    out[f"{b}(p0) vs {a}(p1)"] = res_ba

    return out


def _az_winrate_from_run_match(
    res: Dict[str, Result], az_label: str = "azbsmcts"
) -> float:
    items = list(res.items())
    r1 = items[0][1]
    r2 = items[1][1]
    # r1 label is like "azbsmcts(p0) vs bsmcts(p1)"
    # r2 label is like "bsmcts(p0) vs azbsmcts(p1)"
    az_wins = (
        r1.p0_wins + r2.p1_wins
        if az_label in items[0][0]
        else r1.p1_wins + r2.p0_wins
    )
    total = r1.games + r2.games
    return az_wins / max(1, total)


def _find_run_dirs(runs_root: Path) -> List[Path]:
    return sorted(
        [p for p in runs_root.glob("*") if (p / "model.pt").exists()]
    )


def _load_config(run_dir: Path) -> Dict[str, Any]:
    cfg = run_dir / "config.json"
    return json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}


def _extract_x(run_dir: Path, cfg: Dict[str, Any], x_axis: str) -> float:
    if x_axis == "games":
        if "games" in cfg:
            return float(cfg["games"])
    return float("nan")


def cmd_match(args):
    set_global_seeds(args.seed)
    game = pyspiel.load_game("phantom_go", {"board_size": args.board})

    matchups = [(args.a, args.b)]
    if args.flip_colors:
        matchups.append((args.b, args.a))

    args_dict = dict(vars(args))
    args_dict.pop("func", None)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": args_dict,
        "results": [],
    }

    for a, b in matchups:
        res = run_match(
            game, a, b, args.n, args.T, args.S, args.seed, args.device
        )
        for label, r in res.items():
            summarize(label, r)
            payload["results"].append(
                {
                    "label": label,
                    "games": r.games,
                    "p0_wins": r.p0_wins,
                    "p1_wins": r.p1_wins,
                    "draws": r.draws,
                    "p0_mean_return": r.p0_return_sum / r.games,
                    "p1_mean_return": r.p1_return_sum / r.games,
                }
            )

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote: {outp}")


def cmd_sweep(args):
    import matplotlib.pyplot as plt

    set_global_seeds(args.seed)
    game = pyspiel.load_game("phantom_go", {"board_size": args.board})

    runs_root = Path(args.runs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = _find_run_dirs(runs_root)
    if not run_dirs:
        print(f"No run dirs with model.pt found under {runs_root}")
        return

    xs = []
    wr_vs_bsmcts = []
    wr_vs_random = []

    for rd in run_dirs:
        cfg = _load_config(rd)
        model_path = str(rd / "model.pt")
        os.environ["AZ_MODEL_PATH"] = model_path

        res_b = run_match(
            game,
            "azbsmcts",
            "bsmcts",
            args.n,
            args.T,
            args.S,
            args.seed,
            args.device,
        )
        wr_b = _az_winrate_from_run_match(res_b, az_label="azbsmcts")

        res_r = run_match(
            game,
            "azbsmcts",
            "random",
            args.n,
            args.T,
            args.S,
            args.seed + 999,
            args.device,
        )
        wr_r = _az_winrate_from_run_match(res_r, az_label="azbsmcts")

        x = _extract_x(rd, cfg, args.x_axis)
        xs.append(x)
        wr_vs_bsmcts.append(wr_b)
        wr_vs_random.append(wr_r)

        print(f"{rd.name}: wr_vs_bsmcts={wr_b:.3f} wr_vs_random={wr_r:.3f}")

    order = np.argsort(np.array(xs))
    xs = np.array(xs)[order]
    wr_vs_bsmcts = np.array(wr_vs_bsmcts)[order]
    wr_vs_random = np.array(wr_vs_random)[order]

    plt.figure()
    plt.plot(xs, wr_vs_bsmcts, label="AZ vs BS-MCTS")
    plt.plot(xs, wr_vs_random, label="AZ vs Random")
    plt.xlabel("self-play games (per run)")
    plt.ylabel("win rate")
    plt.ylim(0.0, 1.0)
    plt.title("Evaluation win rate vs training budget")
    plt.legend()
    plt.tight_layout()

    out_path = outdir / "eval_winrate_vs_training.png"
    plt.savefig(out_path)
    plt.close()

    data_path = outdir / "eval_sweep.json"
    payload = [
        {
            "games": float(xs[i]),
            "wr_vs_bsmcts": float(wr_vs_bsmcts[i]),
            "wr_vs_random": float(wr_vs_random[i]),
        }
        for i in range(len(xs))
    ]
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {data_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("match")
    p1.add_argument("--a", type=str, default="azbsmcts")
    p1.add_argument("--b", type=str, default="bsmcts")
    p1.add_argument("--flip-colors", action="store_true")
    p1.add_argument("--n", type=int, default=20)
    p1.add_argument("--T", type=int, default=8)
    p1.add_argument("--S", type=int, default=4)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--device", type=str, default="cpu")
    p1.add_argument("--board", type=int, default=9)
    p1.add_argument("--out-json", type=str, default="")
    p1.set_defaults(func=cmd_match)

    p2 = sub.add_parser("sweep")
    p2.add_argument("--runs", type=str, default="runs")
    p2.add_argument("--outdir", type=str, default="plots")
    p2.add_argument("--x-axis", type=str, default="games", choices=["games"])
    p2.add_argument("--n", type=int, default=20)
    p2.add_argument("--T", type=int, default=8)
    p2.add_argument("--S", type=int, default=4)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--device", type=str, default="cpu")
    p2.add_argument("--board", type=int, default=9)
    p2.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
