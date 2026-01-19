import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyspiel
import torch
import torch.nn.functional as F

from agents import AZBSMCTSAgent
from belief.samplers.particle import (
    ParticleBeliefSampler,
    ParticleDeterminizationSampler,
)
from nets.tiny_policy_value import TinyPolicyValueNet


@dataclass
class Example:
    obs: np.ndarray  # [obs_size]  (SIDE-TO-MOVE perspective)
    pi: np.ndarray  # [num_actions]
    z: float  # scalar target value from SIDE-TO-MOVE perspective


def set_global_seeds(seed: int, deterministic_torch: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def make_run_dir(prefix: str, seed: int) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{ts}_{prefix}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def obs_tensor_side_to_move(state: pyspiel.State) -> np.ndarray:
    side = state.current_player()
    try:
        obs = state.observation_tensor(side)  # if supported
    except TypeError:
        obs = state.observation_tensor()
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def self_play(
    game: pyspiel.Game,
    net: TinyPolicyValueNet,
    num_games: int,
    T: int,
    S: int,
    seed: int,
    temperature: float,
    device: str,
) -> Tuple[List[Example], List[float]]:
    rng = random.Random(seed)

    num_actions = game.num_distinct_actions()
    examples: List[Example] = []
    p0_returns: List[float] = []

    for _ in range(num_games):
        state = game.new_initial_state()

        # Belief samplers per player
        p0 = ParticleBeliefSampler(
            game=game, ai_id=0, seed=rng.randint(0, 10**9)
        )
        p1 = ParticleBeliefSampler(
            game=game, ai_id=1, seed=rng.randint(0, 10**9)
        )

        a0 = AZBSMCTSAgent(
            player_id=0,
            num_actions=num_actions,
            obs_size=len(obs_tensor_side_to_move(state)),
            sampler=ParticleDeterminizationSampler(p0),
            T=T,
            S=S,
            seed=rng.randint(0, 10**9),
            net=net,
            device=device,
        )
        a1 = AZBSMCTSAgent(
            player_id=1,
            num_actions=num_actions,
            obs_size=len(obs_tensor_side_to_move(state)),
            sampler=ParticleDeterminizationSampler(p1),
            T=T,
            S=S,
            seed=rng.randint(0, 10**9),
            net=net,
            device=device,
        )

        traj: List[Tuple[np.ndarray, np.ndarray, int]] = []

        while not state.is_terminal():
            p = state.current_player()
            obs = obs_tensor_side_to_move(state)

            if p == 0:
                action, pi = a0.select_action_with_pi(
                    state, temperature=temperature
                )
            else:
                action, pi = a1.select_action_with_pi(
                    state, temperature=temperature
                )

            traj.append((obs, pi.astype(np.float32), p))

            actor = state.current_player()
            state.apply_action(action)

            # update both particle filters using observation strings
            p0.step(actor=actor, action=action, real_state_after=state)
            p1.step(actor=actor, action=action, real_state_after=state)

        rets = state.returns()
        p0_returns.append(float(rets[0]))

        # SIDE-TO-MOVE value targets: z = return for the player who acted at that state
        for obs, pi, p in traj:
            z = float(rets[p])
            examples.append(Example(obs=obs, pi=pi, z=z))

    return examples, p0_returns


def train_net(
    net: TinyPolicyValueNet,
    examples: List[Example],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    metrics_path=None,
):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    obs = np.stack([ex.obs for ex in examples]).astype(np.float32)
    pi = np.stack([ex.pi for ex in examples]).astype(np.float32)
    z = np.array([ex.z for ex in examples], dtype=np.float32)

    obs_t = torch.from_numpy(obs).to(device)
    pi_t = torch.from_numpy(pi).to(device)
    z_t = torch.from_numpy(z).to(device)

    n = obs.shape[0]

    for ep in range(epochs):
        idx = rng.permutation(n)
        total_loss = total_ploss = total_vloss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            x = obs_t[batch_idx]
            target_pi = pi_t[batch_idx]
            target_z = z_t[batch_idx].unsqueeze(1)

            logits, v = net(x)

            logp = F.log_softmax(logits, dim=1)
            policy_loss = -(target_pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(v, target_z)
            loss = policy_loss + value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_ploss += float(policy_loss.item())
            total_vloss += float(value_loss.item())
            batches += 1

        print(
            f"epoch {ep + 1}/{epochs}  loss={total_loss / batches:.4f}  "
            f"policy={total_ploss / batches:.4f}  value={total_vloss / batches:.4f}"
        )

        if metrics_path is not None:
            rec = {
                "epoch": ep + 1,
                "loss": total_loss / batches,
                "policy_loss": total_ploss / batches,
                "value_loss": total_vloss / batches,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    net.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--S", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="models/model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default="aztrain")
    parser.add_argument("--deterministic-torch", action="store_true")
    parser.add_argument("--logdir", type=str, default="")
    args = parser.parse_args()

    set_global_seeds(args.seed, deterministic_torch=args.deterministic_torch)

    run_dir = (
        Path(args.logdir)
        if args.logdir
        else make_run_dir(args.run_name, args.seed)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_path = run_dir / "train_metrics.jsonl"

    game = pyspiel.load_game("phantom_go", {"board_size": 9})
    tmp = game.new_initial_state()
    obs_size = len(obs_tensor_side_to_move(tmp))
    num_actions = game.num_distinct_actions()

    net = TinyPolicyValueNet(obs_size=obs_size, num_actions=num_actions).to(
        args.device
    )

    print("self-play...")
    examples, p0rets = self_play(
        game=game,
        net=net,
        num_games=args.games,
        T=args.T,
        S=args.S,
        seed=args.seed,
        temperature=args.temp,
        device=args.device,
    )
    print(f"collected examples: {len(examples)}")
    print(f"p0 return mean over self-play games: {np.mean(p0rets):.3f}")

    print("training...")
    train_net(
        net=net,
        examples=examples,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        metrics_path=metrics_path,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_path))
    print(f"saved: {out_path}")

    ckpt_path = run_dir / "model.pt"
    torch.save(net.state_dict(), str(ckpt_path))
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
