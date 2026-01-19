# azbsmcts

![License: Apache 2.0](https://img.shields.io/github/license/huyenngn/azbsmcts)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![CI](https://github.com/huyenngn/azbsmcts/actions/workflows/ci.yml/badge.svg?branch=master)
![Smoke](https://github.com/huyenngn/azbsmcts/actions/workflows/smoke.yml/badge.svg?branch=master)
![Demo model downloads](https://img.shields.io/github/downloads/huyenngn/azbsmcts/demo-model/total?label=demo%20model%20downloads)

AlphaZero-inspired AI agent for **Phantom Go** (imperfect-information Go), developed as part of a research thesis.

The core contribution of this project is **AZ-BS-MCTS**: an AlphaZero-guided Belief-State Monte Carlo Tree Search algorithm that operates under strict partial-information constraints.

For game simulation and environment handling, this project builds upon [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Design Principles

This codebase intentionally enforces the following rules:

- **No access to hidden state**
  - Determinization is performed via belief-state particle filtering.
  - Cloning the real environment state is explicitly disallowed.
- **Explicit player perspectives**
  - `observation_tensor(player_id)` and `observation_string(player_id)` are always used.
  - Ambiguous default observation calls are never used.
- **Clear value semantics**
  - Neural network values are defined from the _side-to-move_ perspective.
  - Values are converted to a fixed root-player perspective during tree backup.
- **Reproducibility over convenience**
  - All scripts accept explicit random seeds.
  - Results are statistically reproducible (not bitwise deterministic), which is standard for MCTS-based methods.

## Demo

Use [uv](https://docs.astral.sh/uv/) to set up a local development environment.

```sh
git clone https://github.com/huyenngn/azbsmcts.git
cd azbsmcts
uv sync
```

You can use `uv run <command>` to avoid having to manually activate the project
venv. For example, to start the backend for the demo, run:

```sh
uv run api
```

For the frontend, follow instructions in [`demo`](demo).

## Scripts Overview

All functionality is exposed via `uv run <script>` commands,
as defined in `pyproject.toml`:

- `api` – Start FastAPI backend for interactive play (demo only)
- `train` – Run self-play training and log results to `runs/`
- `eval` – Offline evaluation (single match or sweep)
- `tune` – Optuna-based hyperparameter tuning
- `plot` – Generate plots from training logs

## Experimental Workflow

### 1. Training

Run self-play training to produce a model checkpoint and logs:

```sh
uv run train \
  --games 50 \
  --T 8 \
  --S 4 \
  --epochs 5 \
  --seed 0 \
  --out models/model.pt
```

Each training run creates a directory under `runs/` containing:

- `config.json` – full run configuration
- `train_metrics.jsonl` – per-epoch training losses
- `model.pt` – trained network checkpoint

---

### 2. Hyperparameter Tuning

Hyperparameter tuning uses Optuna.
Each trial performs a short training run followed by evaluation against a
BS-MCTS baseline.

```sh
uv run tune \
  --trials 200 \
  --games 500 \
  --epochs 5 \
  --eval-n 20 \
  --storage sqlite:///runs/optuna.db \
  --study-name az9x9
```

Each trial produces its own directory under `runs/`, containing:

- `config.json`
- `train_metrics.jsonl`
- `model.pt`
- `eval.json`

Best trial summary is written to `runs/optuna_best.json`

---

### 3. Evaluation

Evaluation is handled by a single script with two modes.

#### Single Match Evaluation

Evaluate one trained model against a baseline:

```sh
AZ_MODEL_PATH=models/model.pt \
uv run eval match \
  --a azbsmcts \
  --b bsmcts \
  --n 20 \
  --T 8 \
  --S 4 \
  --seed 0 \
  --out-json runs/eval_seed0.json
```

This produces JSON suitable for tables and statistical analysis.

#### Sweep Evaluation

Evaluate multiple trained checkpoints (e.g. across training budgets):

```sh
uv run eval sweep \
  --runs runs \
  --n 20 \
  --T 8 \
  --S 4
```

This generates:

- `plots/eval_winrate_vs_training.png`
- `plots/eval_sweep.json`

---

### 4. Plotting

Training metrics are logged as JSONL files `runs/*/train_metrics.jsonl`

Generate per-run loss curves:

```sh
uv run plot
```

Generate per-run plots plus a combined loss plot:

```sh
uv run plot --combined
```

Plots are written to the `plots/` directory.

## Reproducibility

All scripts support explicit random seeds. Seeds are applied to:

- Python's `random`
- NumPy
- PyTorch

Due to stochastic belief sampling and Monte Carlo Tree Search, results are statistically reproducible but not bitwise deterministic. This is expected and appropriate for MCTS-based research.

## License

This project is licensed under the Apache License 2.0. For the full license text, see the [`LICENSE`](LICENSE) file.

It contains modifications of [OpenSpiel's](https://github.com/google-deepmind/open_spiel) AlphaZero algorithm and MCTS implementations, originally developed by Google DeepMind. The original license has been preserved in the relevant source files.
