from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import typing as t

import numpy as np

import openspiel
from scripts.common import config, io, seeding
from scripts.eval import match

OPPONENTS = ["bsmcts", "random"]


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


def extract_az_game_lengths(
  res: dict, az_label: str = "azbsmcts"
) -> tuple[list[int], list[int], list[int]]:
  """Extract win/loss/draw game lengths from AZ perspective.

  Returns:
      (win_lengths, loss_lengths, draw_lengths)
  """
  items = list(res.items())
  r1 = items[0][1]
  r2 = items[1][1]

  if az_label in items[0][0]:
    # AZ is p0 in r1, p1 in r2
    win_lengths = r1.win_lengths + r2.loss_lengths
    loss_lengths = r1.loss_lengths + r2.win_lengths
    draw_lengths = r1.draw_lengths + r2.draw_lengths
  else:
    # AZ is p1 in r1, p0 in r2
    win_lengths = r1.loss_lengths + r2.win_lengths
    loss_lengths = r1.win_lengths + r2.loss_lengths
    draw_lengths = r1.draw_lengths + r2.draw_lengths

  return win_lengths, loss_lengths, draw_lengths


def mean_or_nan(values: list[int]) -> float:
  """Compute mean of list or NaN if empty."""
  return float(np.mean(values)) if values else float("nan")


def find_checkpoints(run_dir: pathlib.Path) -> list[pathlib.Path]:
  """Find all checkpoint files in a run directory."""
  checkpoints = []
  # Main model
  if (run_dir / "model.pt").exists():
    checkpoints.append(run_dir / "model.pt")
  # Intermediate checkpoints
  ckpt_dir = run_dir / "checkpoints"
  if ckpt_dir.exists():
    checkpoints.extend(sorted(ckpt_dir.glob("checkpoint_games_*.pt")))
  return checkpoints


def extract_games_from_checkpoint(path: pathlib.Path) -> int | None:
  """Extract game count from checkpoint filename."""
  name = path.stem
  if name == "model":
    return None  # Will use config.json
  if name.startswith("checkpoint_games_"):
    try:
      return int(name.replace("checkpoint_games_", ""))
    except ValueError:
      return None
  return None


def build_signature(
  *,
  args: argparse.Namespace,
  game_cfg: config.GameConfig,
  run_dir: pathlib.Path,
) -> dict[str, t.Any]:
  """Build deterministic signature for resume compatibility checks."""
  return {
    "run_dir": str(run_dir.resolve()),
    "game": {
      "name": game_cfg.name,
      "params": game_cfg.params,
    },
    "n": args.n,
    "seed": args.seed,
    "device": args.device,
    "search": {
      "T": args.T,
      "S": args.S,
      "c_puct": args.c_puct,
      "dirichlet_alpha": args.dirichlet_alpha,
      "dirichlet_weight": args.dirichlet_weight,
    },
    "sampler": {
      "num_particles": args.num_particles,
      "matches_per_particle": args.matches_per_particle,
      "rebuild_tries": args.rebuild_tries,
    },
    "opponents": OPPONENTS,
  }


def completed_checkpoints(rows: list[dict[str, t.Any]]) -> set[str]:
  """Collect completed checkpoint IDs from existing rows."""
  done: set[str] = set()
  for row in rows:
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, str):
      done.add(checkpoint)
  return done


def load_meta_signature(path: pathlib.Path) -> dict[str, t.Any] | None:
  """Load signature from metadata file if valid, else return None."""
  if not path.exists():
    return None

  try:
    signature = io.read_json(path)
  except Exception:
    return None

  if not isinstance(signature, dict):
    return None

  return signature


def apply_signature_to_args(
  args: argparse.Namespace, signature: dict[str, t.Any]
) -> bool:
  """Apply saved signature values onto parsed CLI args."""
  try:
    game = signature["game"]
    search = signature["search"]
    sampler = signature["sampler"]

    args.game = str(game["name"])
    args.game_params = json.dumps(game["params"], sort_keys=True)

    args.n = int(signature["n"])
    args.seed = int(signature["seed"])
    args.device = str(signature["device"])

    args.T = int(search["T"])
    args.S = int(search["S"])
    args.c_puct = float(search["c_puct"])
    args.dirichlet_alpha = float(search["dirichlet_alpha"])
    args.dirichlet_weight = float(search["dirichlet_weight"])

    args.num_particles = int(sampler["num_particles"])
    args.matches_per_particle = int(sampler["matches_per_particle"])
    args.rebuild_tries = int(sampler["rebuild_tries"])
    return True
  except (KeyError, TypeError, ValueError):
    return False


def append_row(path: pathlib.Path, row: dict[str, t.Any]) -> None:
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", type=str, default="cpu")

  p.add_argument("--game", type=str, default="phantom_go")
  p.add_argument("--game-params", type=str, default='{"board_size": 5}')

  p.add_argument("--n", type=int, default=20)
  p.add_argument("--T", type=int, default=8)
  p.add_argument("--S", type=int, default=4)
  p.add_argument("--c-puct", type=float, default=1.5)
  p.add_argument("--dirichlet-alpha", type=float, default=0.0)
  p.add_argument("--dirichlet-weight", type=float, default=0.0)

  p.add_argument(
    "--run-dir",
    type=str,
    required=True,
    help="Path to a single training run directory (e.g., runs/run_20260130_123456)",
  )
  p.add_argument(
    "--resume",
    action="store_true",
    help="Resume eval sweep by skipping checkpoints already in eval_sweep.jsonl",
  )

  p.add_argument("--x-axis", type=str, default="games")  # from run config

  p.add_argument("--num-particles", type=int, default=50)
  p.add_argument("--matches-per-particle", type=int, default=15)
  p.add_argument("--rebuild-tries", type=int, default=5)

  args = p.parse_args()

  run_dir = pathlib.Path(args.run_dir)
  if not run_dir.exists():
    print(f"Run directory not found: {run_dir}")
    return
  if not (run_dir / "model.pt").exists():
    print(f"No model.pt found in {run_dir}")
    return

  out_path = run_dir / "eval_sweep.jsonl"
  meta_path = run_dir / "eval_sweep.meta.json"

  if args.resume:
    saved_signature = load_meta_signature(meta_path)
    if saved_signature is not None:
      if apply_signature_to_args(args, saved_signature):
        print(f"[resume] Using saved config from {meta_path}")
      else:
        print(
          "[resume] Warning: metadata signature is invalid; "
          "falling back to CLI args"
        )
    else:
      print(
        f"[resume] Warning: no valid metadata found at {meta_path}; "
        "using CLI args"
      )

  game_cfg = config.GameConfig.from_cli(args.game, args.game_params)
  search_cfg = config.SearchConfig(
    T=args.T,
    S=args.S,
    c_puct=args.c_puct,
    dirichlet_alpha=args.dirichlet_alpha,
    dirichlet_weight=args.dirichlet_weight,
  )
  sampler_cfg = config.SamplerConfig(
    num_particles=args.num_particles,
    matches_per_particle=args.matches_per_particle,
    rebuild_tries=args.rebuild_tries,
  )

  game = openspiel.Game(game_cfg.name, game_cfg.params)

  cfg_path = run_dir / "config.json"
  cfg = io.read_json(cfg_path) if cfg_path.exists() else {}

  # Get total games from config for final model
  total_games = float("nan")
  if isinstance(cfg, dict):
    if "config" in cfg and isinstance(cfg["config"], dict):
      inner = cfg["config"]
      if "games" in inner:
        total_games = float(inner["games"])
      elif (
        "budget" in inner
        and isinstance(inner["budget"], dict)
        and "games" in inner["budget"]
      ):
        total_games = float(inner["budget"]["games"])
    elif "games" in cfg:
      total_games = float(cfg["games"])
    elif (
      "budget" in cfg
      and isinstance(cfg["budget"], dict)
      and "games" in cfg["budget"]
    ):
      total_games = float(cfg["budget"]["games"])

  # Find all checkpoints (intermediate + final)
  checkpoints = find_checkpoints(run_dir)
  if not checkpoints:
    print(f"No checkpoints found in {run_dir}")
    return

  signature = build_signature(args=args, game_cfg=game_cfg, run_dir=run_dir)

  done_checkpoints: set[str] = set()

  if args.resume:
    if out_path.exists():
      existing_rows = io.read_jsonl(out_path)
      done_checkpoints = completed_checkpoints(existing_rows)
  else:
    out_path.write_text("", encoding="utf-8")

  io.write_json(meta_path, signature)

  pending_checkpoints = [
    ckpt for ckpt in checkpoints if ckpt.name not in done_checkpoints
  ]

  print(f"discovered checkpoints: {len(checkpoints)}")
  print(f"pending checkpoints: {len(pending_checkpoints)}")

  sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  for ckpt_path in pending_checkpoints:
    # Determine games count for this checkpoint
    games_from_name = extract_games_from_checkpoint(ckpt_path)
    x_games = games_from_name if games_from_name else total_games

    model_path = str(ckpt_path)
    ckpt_id = ckpt_path.name

    # Evaluate vs BS-MCTS
    res_b = match.run_match(
      game=game,
      a="azbsmcts",
      b="bsmcts",
      n=args.n,
      search_cfg=search_cfg,
      sampler_cfg=sampler_cfg,
      seed=args.seed,
      device=args.device,
      model_path=model_path,
      run_id=f"sweep_{sweep_id}_{ckpt_id}",
    )
    wr_b = az_winrate(res_b)
    win_b, loss_b, draw_b = extract_az_game_lengths(res_b)

    # Evaluate vs random
    random_seed = seeding.derive_seed(
      args.seed, purpose="eval/sweep_random", extra=ckpt_id
    )
    res_r = match.run_match(
      game=game,
      a="azbsmcts",
      b="random",
      n=args.n,
      search_cfg=search_cfg,
      sampler_cfg=sampler_cfg,
      seed=random_seed,
      device=args.device,
      model_path=model_path,
      run_id=f"sweep_{sweep_id}_{ckpt_id}_random",
    )
    wr_r = az_winrate(res_r)
    win_r, loss_r, draw_r = extract_az_game_lengths(res_r)

    row = {
      "checkpoint": ckpt_path.name,
      "model_path": model_path,
      "x_games": float(x_games) if x_games else float("nan"),
      "wr_vs_bsmcts": float(wr_b),
      "wr_vs_random": float(wr_r),
      "mean_win_len_vs_bsmcts": mean_or_nan(win_b),
      "mean_loss_len_vs_bsmcts": mean_or_nan(loss_b),
      "mean_draw_len_vs_bsmcts": mean_or_nan(draw_b),
      "mean_win_len_vs_random": mean_or_nan(win_r),
      "mean_loss_len_vs_random": mean_or_nan(loss_r),
      "mean_draw_len_vs_random": mean_or_nan(draw_r),
    }
    append_row(out_path, row)
    print(
      f"{ckpt_id}: games={x_games} "
      f"wr_vs_bsmcts={wr_b:.3f} wr_vs_random={wr_r:.3f}"
    )

  print(f"Wrote {out_path}")


if __name__ == "__main__":
  main()
