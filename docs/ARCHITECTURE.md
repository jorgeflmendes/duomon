# DuoMon Architecture

This document describes the runtime, training, and analysis boundaries for the
multi-agent battle-coordination system.

## Runtime Decision Flow

```mermaid
sequenceDiagram
    participant Showdown
    participant Adapter as Battle adapter
    participant A as Slot agent A
    participant B as Slot agent B
    participant Joint as Joint selector
    participant Policy as CTDE reranker

    Showdown->>Adapter: current battle state
    Adapter->>A: local observation and legal actions
    Adapter->>B: local observation and legal actions
    A-->>Joint: ranked proposal and intent message
    B-->>Joint: ranked proposal and intent message
    Joint->>Policy: candidate joint actions and features
    Policy-->>Joint: reranked joint choice
    Joint-->>Showdown: coordinated actions
```

The execution-time contract is decentralized at the slot-agent boundary:
agents retain local observations and communicate structured intent rather than
sharing unrestricted internal state.

## Training and Evaluation Loop

```mermaid
flowchart LR
    BENCH["Benchmark battles"] --> TRACES["Decision traces<br/>JSONL"]
    BENCH --> REPLAYS["Battle replays"]
    TRACES --> FEATURES["Joint-action feature extraction"]
    FEATURES --> TRAIN["CTDE training"]
    TRAIN --> MODEL["Reranker artifacts"]
    MODEL --> RUNTIME["Runtime profile"]
    RUNTIME --> BENCH
    TRACES --> REPORTS["Metrics and analysis reports"]
    REPLAYS --> DASH["Dashboard"]
    REPORTS --> DASH
```

This loop separates data collection, feature extraction, model training,
runtime inference, and visual inspection. It keeps experimental artifacts under
explicit output paths instead of coupling them to package code.

## Component Responsibilities

| Component | Responsibility | Boundary |
| --- | --- | --- |
| `duomon/battle` | Adapts Pokemon Showdown state into internal observations | Does not choose actions |
| `duomon/policy_core` | Legal action generation and independent policy primitives | Does not own coordination |
| `duomon/agents` | Intent exchange and joint action selection | Does not train models |
| `duomon/heuristic` | Tactical scoring and threat estimation | Provides features, not final authority |
| `duomon/ctde` | Feature builders, models, training, and evaluation | Produces artifacts consumed by runtime profiles |
| `duomon/benchmarking` | Battle orchestration, metrics, traces, and reports | Does not implement model internals |
| `web` | Local dashboard over saved traces, replays, and metrics | Read-only analysis surface |

## Dashboard Boundary

```mermaid
flowchart TB
    OUTPUTS["Experiment outputs"] --> SERVER["web/demo_server.py"]
    SERVER --> INDEX["Dashboard UI"]
    INDEX --> MATCH["Match summary"]
    INDEX --> TURN["Per-turn reasoning"]
    INDEX --> REPLAY["Replay links"]
```

The dashboard reads generated artifacts and does not control live battles. This
keeps visualization failures from affecting experiment execution.

## Architectural Constraints

- Full runtime validation requires local Pokemon Showdown server and client
  submodules.
- CI currently checks syntax portability; it does not simulate battles.
- Reported results depend on teams, opponents, seeds, and model artifacts.
- Model outputs are experimental signals and should not be treated as robust
  competitive ladder performance.
