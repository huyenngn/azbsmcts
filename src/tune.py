import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pyspiel
import torch

from eval import run_match
from nets.tiny_policy_value import TinyPolicyValueNet
from train import self_play, set_global_seeds, train_net


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--board", type=int, default=9)

    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)

    parser.add_argument("--eval-n", type=int, default=10)

    parser.add_argument("--study-name", type=str, default="az_tune")
    parser.add_argument(
        "--storage", type=str, default="sqlite:///runs/optuna.db"
    )
    parser.add_argument("--direction", type=str, default="maximize")
    args = parser.parse_args()

    ensure_dir(Path("runs"))
    set_global_seeds(args.seed, deterministic_torch=False)

    game = pyspiel.load_game("phantom_go", {"board_size": args.board})
    tmp = game.new_initial_state()
    obs_size = len(tmp.observation_tensor())
    num_actions = game.num_distinct_actions()

    # FIXED ARCHITECTURE (keep tuning simple and checkpoints compatible)
    FIXED_HIDDEN = 256

    def objective(trial: optuna.Trial) -> float:
        # Search hyperparameters (keep small + stable)
        T = trial.suggest_int("T", 2, 16, log=True)
        S = trial.suggest_int("S", 2, 8)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        temp = trial.suggest_float("temp", 0.5, 1.5)
        c_puct = trial.suggest_float("c_puct", 0.5, 3.0, log=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("runs") / f"{ts}_trial{trial.number:04d}"
        ensure_dir(run_dir)

        cfg = {
            "trial": trial.number,
            "params": dict(trial.params),
            "budget": {
                "games": args.games,
                "epochs": args.epochs,
                "batch": args.batch,
                "eval_n": args.eval_n,
            },
            "seed": args.seed,
            "device": args.device,
            "board": args.board,
            "arch": {"hidden": FIXED_HIDDEN},
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        net = TinyPolicyValueNet(
            obs_size=obs_size,
            num_actions=num_actions,
            hidden=FIXED_HIDDEN,
        ).to(args.device)

        examples, p0rets = self_play(
            game=game,
            net=net,
            num_games=args.games,
            T=T,
            S=S,
            seed=args.seed + trial.number * 1_000_000,
            temperature=temp,
            device=args.device,
        )

        metrics_path = run_dir / "train_metrics.jsonl"
        train_net(
            net=net,
            examples=examples,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=lr,
            device=args.device,
            seed=args.seed + trial.number * 1_000_000,
            metrics_path=metrics_path,
        )

        model_path = run_dir / "model.pt"
        torch.save(net.state_dict(), str(model_path))

        # Point eval at this trial model
        os.environ["AZ_MODEL_PATH"] = str(model_path)
        os.environ["AZ_DEVICE"] = args.device

        # Ensure eval uses same MCTS constants as training
        os.environ["AZ_C_PUCT"] = str(c_puct)

        # Evaluate vs baseline
        res = run_match(
            game,
            a="azbsmcts",
            b="bsmcts",
            n=args.eval_n,
            T=T,
            S=S,
            seed=args.seed + 42 + trial.number * 99,
            device=args.device,
        )

        items = list(res.items())
        r1 = items[0][1]  # az as p0
        r2 = items[1][1]  # az as p1
        az_wins = r1.p0_wins + r2.p1_wins
        total = r1.games + r2.games
        winrate = az_wins / max(1, total)

        with (run_dir / "eval.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "winrate_az_vs_bsmcts": float(winrate),
                    "p0_return_mean_selfplay": float(np.mean(p0rets))
                    if p0rets
                    else None,
                    "arch_hidden": FIXED_HIDDEN,
                },
                f,
                indent=2,
            )

        trial.set_user_attr("run_dir", str(run_dir))
        return float(winrate)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
    )
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_trial": best.number,
        "best_run_dir": best.user_attrs.get("run_dir"),
        "arch": {"hidden": FIXED_HIDDEN},
    }
    (Path("runs") / "optuna_best.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nBest trial:")
    print(summary)


if __name__ == "__main__":
    main()
