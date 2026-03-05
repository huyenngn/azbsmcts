from __future__ import annotations

import argparse
import json
import sys
import typing as t
from typing import TYPE_CHECKING

from scripts.common import config, io
from scripts.eval import match, sweep

if TYPE_CHECKING:
  import pathlib


def _prepare_run_dir(
  tmp_path: pathlib.Path, checkpoints: list[int]
) -> pathlib.Path:
  run_dir = tmp_path / "run"
  ckpt_dir = run_dir / "checkpoints"
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  (run_dir / "model.pt").touch()
  for games in checkpoints:
    (ckpt_dir / f"checkpoint_games_{games:05d}.pt").touch()

  (run_dir / "config.json").write_text(
    json.dumps({"config": {"budget": {"games": 1000}}}),
    encoding="utf-8",
  )
  return run_dir


def _dummy_row(checkpoint: str) -> dict[str, t.Any]:
  return {
    "checkpoint": checkpoint,
    "model_path": checkpoint,
    "x_games": 1000.0,
    "wr_vs_bsmcts": 0.5,
    "wr_vs_random": 0.5,
    "mean_win_len_vs_bsmcts": 10.0,
    "mean_loss_len_vs_bsmcts": 10.0,
    "mean_draw_len_vs_bsmcts": 10.0,
    "mean_win_len_vs_random": 10.0,
    "mean_loss_len_vs_random": 10.0,
    "mean_draw_len_vs_random": 10.0,
  }


def _signature(
  run_dir: pathlib.Path, *, n: int = 20, seed: int = 0
) -> dict[str, t.Any]:
  args = argparse.Namespace(
    run_dir=str(run_dir),
    game="phantom_go",
    game_params='{"board_size": 5}',
    n=n,
    seed=seed,
    device="cpu",
    T=8,
    S=4,
    c_puct=1.5,
    dirichlet_alpha=0.0,
    dirichlet_weight=0.0,
    num_particles=50,
    matches_per_particle=15,
    rebuild_tries=5,
  )
  game_cfg = config.GameConfig.from_cli(args.game, args.game_params)
  return sweep.build_signature(
    args=args,
    game_cfg=game_cfg,
    run_dir=run_dir,
  )


def _patch_runtime(
  monkeypatch: t.Any, call_log: list[dict[str, t.Any]]
) -> None:
  monkeypatch.setattr(
    sweep.openspiel,
    "Game",
    lambda _name, _params: object(),
  )

  def fake_run_match(
    *,
    game: object,
    a: str,
    b: str,
    n: int,
    search_cfg: config.SearchConfig,
    sampler_cfg: config.SamplerConfig,
    seed: int,
    device: str,
    model_path: str | None,
    run_id: str,
  ) -> dict[str, match.Result]:
    del game, search_cfg, sampler_cfg, seed, device, run_id
    call_log.append(
      {
        "a": a,
        "b": b,
        "n": n,
        "model_path": model_path,
      }
    )
    r1 = match.Result(
      games=n,
      p0_wins=n,
      p1_wins=0,
      draws=0,
      win_lengths=[7] * n,
      loss_lengths=[],
      draw_lengths=[],
    )
    r2 = match.Result(
      games=n,
      p0_wins=0,
      p1_wins=n,
      draws=0,
      win_lengths=[],
      loss_lengths=[8] * n,
      draw_lengths=[],
    )
    return {
      f"{a}(p0) vs {b}(p1)": r1,
      f"{b}(p0) vs {a}(p1)": r2,
    }

  monkeypatch.setattr(sweep.match, "run_match", fake_run_match)


def _run_main(
  monkeypatch: t.Any,
  run_dir: pathlib.Path,
  extra_args: list[str],
) -> None:
  argv = ["eval-sweep", "--run-dir", str(run_dir), *extra_args]
  monkeypatch.setattr(sys, "argv", argv)
  sweep.main()


def test_resume_skips_completed_checkpoints(
  tmp_path: pathlib.Path, monkeypatch: t.Any
) -> None:
  run_dir = _prepare_run_dir(tmp_path, checkpoints=[100, 300])
  out_path = run_dir / "eval_sweep.jsonl"
  meta_path = run_dir / "eval_sweep.meta.json"

  done_ckpt = "checkpoint_games_00100.pt"
  io.write_jsonl(out_path, [_dummy_row(done_ckpt)])
  io.write_json(meta_path, _signature(run_dir, n=2))

  calls: list[dict[str, t.Any]] = []
  _patch_runtime(monkeypatch, calls)
  _run_main(monkeypatch, run_dir, ["--resume", "--n", "2"])

  rows = io.read_jsonl(out_path)
  assert len(rows) == 3
  assert {r["checkpoint"] for r in rows} == {
    "model.pt",
    "checkpoint_games_00100.pt",
    "checkpoint_games_00300.pt",
  }

  assert len(calls) == 4
  skipped_path = str(run_dir / "checkpoints" / done_ckpt)
  assert all(c["model_path"] != skipped_path for c in calls)


def test_resume_uses_saved_config_on_mismatch(
  tmp_path: pathlib.Path, monkeypatch: t.Any
) -> None:
  run_dir = _prepare_run_dir(tmp_path, checkpoints=[100])
  out_path = run_dir / "eval_sweep.jsonl"
  meta_path = run_dir / "eval_sweep.meta.json"

  io.write_jsonl(out_path, [_dummy_row("checkpoint_games_00100.pt")])
  io.write_json(meta_path, _signature(run_dir, n=9))

  calls: list[dict[str, t.Any]] = []
  _patch_runtime(monkeypatch, calls)

  _run_main(monkeypatch, run_dir, ["--resume", "--n", "2"])

  assert len(calls) == 2
  assert all(c["n"] == 9 for c in calls)
  rows = io.read_jsonl(out_path)
  assert {r["checkpoint"] for r in rows} == {
    "model.pt",
    "checkpoint_games_00100.pt",
  }


def test_auto_discovery_evaluates_model_and_final_checkpoint(
  tmp_path: pathlib.Path, monkeypatch: t.Any
) -> None:
  run_dir = _prepare_run_dir(tmp_path, checkpoints=[1000])
  out_path = run_dir / "eval_sweep.jsonl"

  calls: list[dict[str, t.Any]] = []
  _patch_runtime(monkeypatch, calls)
  _run_main(monkeypatch, run_dir, ["--n", "1"])

  rows = io.read_jsonl(out_path)
  assert [r["checkpoint"] for r in rows] == [
    "model.pt",
    "checkpoint_games_01000.pt",
  ]
  assert len(calls) == 4


def test_fresh_run_truncates_old_output_and_rewrites_meta(
  tmp_path: pathlib.Path, monkeypatch: t.Any
) -> None:
  run_dir = _prepare_run_dir(tmp_path, checkpoints=[100])
  out_path = run_dir / "eval_sweep.jsonl"
  meta_path = run_dir / "eval_sweep.meta.json"

  io.write_jsonl(out_path, [_dummy_row("old.pt")])
  io.write_json(meta_path, _signature(run_dir, n=9))

  calls: list[dict[str, t.Any]] = []
  _patch_runtime(monkeypatch, calls)
  _run_main(monkeypatch, run_dir, ["--n", "2"])

  rows = io.read_jsonl(out_path)
  assert len(rows) == 2
  assert {r["checkpoint"] for r in rows} == {
    "model.pt",
    "checkpoint_games_00100.pt",
  }

  signature = io.read_json(meta_path)
  assert signature == _signature(run_dir, n=2)
