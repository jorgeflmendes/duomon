# DuoMon

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Node.js 20+](https://img.shields.io/badge/Node.js-20%2B-green)](https://nodejs.org/)

DuoMon is a cooperative multi-agent controller for Generation 9 Pokemon Showdown multi battles. Two allied slot agents act independently at execution time, exchange structured intent messages, and use a CTDE-trained joint-action reranker to choose coordinated turns. 

The project is built for reproducible agent-reasoning experiments: local Showdown simulation, configurable opponents, replay capture, benchmark metrics, and a dashboard for inspecting both match outcomes and per-turn agent decisions.

## Highlights

- Cooperative multi-agent architecture with independent slot-agent execution.
- CTDE-trained joint-action reranker using structured intent messages.
- Deterministic local Showdown simulation with configurable benchmark opponents.
- Support for generalization evaluation using randomly sampled ally teams.
- Built-in dashboard for tracing per-turn agent decisions and analyzing battle replays.

## Tech Stack

- Python 3.11+
- Node.js 20+
- PyTorch (for CTDE training)
- Pokemon Showdown Server & Client

## Repository Layout

- `duomon/`: Core agent logic, battle adapters, and benchmarking framework.
  - `agents/`: Slot-agent runtime, communication, joint selection.
  - `battle/`: Multi-battle Showdown state adapters.
  - `benchmarking/`: Benchmark runner, metrics, result collection.
  - `core/`: Shared filesystem, JSONL, profiling and team utilities.
  - `ctde/`: CTDE MLP data, features, training and evaluation.
  - `heuristic/`: Tactical scoring, threat estimation, damage utilities.
  - `opponents/`: Benchmark opponents.
  - `policy_core/`: Legal action generation and independent slot scoring.
- `experiments/`: Transformer pretraining experiment sandbox.
- `scripts/`: Dataset builders, dependency setups, and CTDE training entrypoints.
- `web/`: Browser-side dashboard modules, backend helpers, and CSS modules.
- `teams/`: Curated ally teams for fixed benchmarks.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 20+
- npm
- Git

### 2. Clone and Submodules

```bash
git clone --recurse-submodules <repo-url>
cd Autonomous-Agents-And-Multi-Agent-Systems-26
```

*(If cloned without submodules, run: `git submodule update --init --recursive`)*

### 3. Install Dependencies & Build

```bash
python -m pip install -r requirements.txt
python scripts/setup/external_dependencies.py --build
```

### 4. Start the Dashboard

```bash
python web/demo_server.py
```
*(On Windows you can also use `start_demo.bat`)*

Then open your browser at `http://127.0.0.1:8765/`.

## Configuration Model

Runtime configuration is defined via environment variables.

Main variables:

- `DUOMON_PROFILE`: Runtime profile (`ctde_mlp` by default).
- `DUOMON_BATTLES_PER_OPPONENT`: Amount of battles to run per opponent (`100` by default).
- `DUOMON_OUTPUT_DIR`: Directory for benchmark outputs and replays.
- `DUOMON_MODEL_DIR`: Directory containing generated CTDE training artifacts.
- `DUOMON_FIXED_ALLY_TEAMS`: Set to `1` to use fixed curated ally teams, or `0` for random.
- `DUOMON_PARALLEL_BATTLES`: Maximum number of concurrent benchmark battles (`32` by default).
- `DUOMON_AUTOSTART_SHOWDOWN`: Set to `1` to let the runner start Showdown automatically.

## Benchmarks & Commands

Default benchmark profile:

```bash
python -m duomon
```

Fixed ally-team benchmark with explicit profile:

```bash
DUOMON_FIXED_ALLY_TEAMS=1 DUOMON_PROFILE=ctde_mlp python -m duomon
```

Small smoke test targeting specific opponents:

```bash
python -m duomon --profile ctde_mlp --opponents simpleheuristics,abyssal --battles 5
```

## Training Workflow

CTDE training is offline and uses the decision traces produced by previous benchmarks.

1. **Collect Traces:** Run a benchmark to gather candidate-pairs and terminal outcomes.
   ```bash
   DUOMON_PROFILE=ctde_mlp python -m duomon --opponents simpleheuristics,abyssal --battles 100
   ```
2. **Train Models:** Run the CTDE training script on the generated logs.
   ```bash
   python scripts/training/train_ctde.py --opponent both
   ```
3. **Evaluate:** Run a new benchmark using the trained profile to evaluate the resulting reranker.
   ```bash
   DUOMON_PROFILE=ctde_mlp python -m duomon --opponents simpleheuristics,abyssal --battles 100
   ```

## Current Results

Latest validated local runs, using the `ctde_mlp` profile with structured communication enabled:

| Mode | Battles | Finished | Errors | Overall WR | Random | MaxPower | Simple | Abyssal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed ally teams | 800 | 800 | 0 | 87.0% | 99.5% | 94.5% | 78.0% | 76.0% |
| Random ally teams | 800 | 793 | 7 | 75.9% | 97.0% | 89.0% | 58.5% | 58.5% |
