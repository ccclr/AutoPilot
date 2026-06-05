# AutoPilot

AutoPilot combines the **Autobahn consensus protocol** with **online CMAB tuning**. Rust runs the consensus stack; Python handles metrics collection and adaptive parameter control.

The consensus layer is based on [Autobahn](https://github.com/neilgiri/autobahn-artifact). For benchmark deployment details inherited from Narwhal, see [benchmark/README.md](benchmark/README.md).

## Layout

```
AutoPilot/
├── node/       primary, worker, benchmark_client
├── primary/    DAG + Autobahn consensus (core.rs)
├── worker/     transaction batching
├── config/     Parameters
├── Agent/      metrics_collector + CMAB (rl/)
└── benchmark/  Fabric local/remote testbed
```

## Setup & Run

```bash
./install_deps.sh
pip install -r benchmark/requirements.txt
pip install numpy scikit-learn joblib psutil
```

**Recommended** — one-command local benchmark with the full learning loop:

```bash
cd benchmark && fab local
```

This builds `node` with the `benchmark` feature, generates config files, and launches primaries, workers, metrics collector, RL controller, and clients via tmux. Tune `bench_params` and `node_params` in `benchmark/fabfile.py`.

**Manual** (single node):

```bash
cargo build --release --features benchmark   # from node/

# Primary
RUST_STATE_SOCKET_PATH=/tmp/autobahn_core_0.sock \
  ./target/release/node -vv run --keys .node-0.json --committee .committee.json \
  --store .db-0 --parameters .parameters-0.json primary

# Worker
./target/release/node -vv run --keys .node-0.json --committee .committee.json \
  --store .db-0-0 --parameters .parameters-0.json worker --id 0

# Metrics + learning
python3 Agent/metrics_collector.py 65 10 --node-index 0 --log-dir ./logs --parameters-file .parameters-0.json
python3 Agent/rl/train_cmab_continuous.py --metrics-dir ./metrics-0 --parameters-file ./.parameters.json --node-index 0
```

Key paths: `RUST_STATE_SOCKET_PATH` (core ↔ collector), `/tmp/autobahn_rl_param_{i}.sock` (CMAB → core), `metrics-{i}/global_state_epoch_{N}.json` (CMAB input), `.parameters.json` (hot-reloaded at `applied_begin`).

## Architecture

**Rust:** `worker` batches transactions → `primary` builds the DAG and runs Autobahn consensus in `core.rs` → commits trigger metrics collection, state aggregation, and parameter application.

**Python:** `metrics_collector` parses logs and exchanges state with core over a Unix socket. `train_cmab_continuous.py` (via `AutobahnController` in benchmarks) reads aggregated global state, selects consensus parameters, writes `.parameters.json`, and notifies core.

## Consensus (Autobahn)

Implemented in `primary/src/core.rs`. Each slot follows Prepare → (optional Confirm) → Commit. Fast path skips Confirm when all replicas vote Prepare in time; slow path requires 2f+1 at each stage.

Notable knobs (several are RL-tunable): `k` (parallel open instances), `cut_condition_type` (tip threshold), `use_fast_path` / `fast_path_timeout`, `use_optimistic_tips`, `epoch_slots` / `window_size` (metrics window), `applied_begin` (slot to apply RL params).

On commit, nodes aggregate `StateReport`s into `global_state_epoch_{epoch}.json` with `global_reward`, `global_fast_path_ratio`, and lane growth rates. Per-node reward: `1000 / (latency_ms + 1)`; `global_reward` is the committee median or mean.

## Learning (CMAB)

Production path is **CMAB**, not PPO (`Agent/rl/envs/` is experimental only).

```
global_state_epoch_t  →  ContextBuilder  →  CMABPolicy.select_arm (RF Thompson Sampling)
       →  ArmCatalog.decode  →  .parameters.json + socket  →  core applies at applied_begin
       →  next epoch  →  global_state_epoch_{t+1}  →  reward  →  policy.update
```

Action at epoch *t* is credited with reward from epoch *t+1* (`global_reward`).

| Piece | Role |
|-------|------|
| `cmab/context_builder.py` | Context from lane growth rates + fast-path ratio |
| `cmab/arm_catalog.py` | Discrete arm space over 5 consensus params |
| `cmab/policy.py` | RandomForest + bootstrap Thompson Sampling |
| `cmab/trainer.py` | Main loop: wait → select → apply → observe → update |
| `actions/action_encode.py` | Arm ↔ parameter mapping |
| `controllers/controller.py` | Spawns CMAB subprocess during benchmarks |

**Arm space:** `batch_size` {100k, 500k}, `header_size` {32, 64}, `cut_condition_type` {1, 3, 4}, `fast_path_timeout` {0, 100, 300} ms, `k` {1, 4}.

```bash
python3 Agent/rl/train_cmab_continuous.py \
  --metrics-dir ./metrics-0 --parameters-file ./.parameters.json \
  --policy rf_ts --context-mode dynamic --node-index 0
```

## License

[Apache 2.0](LICENSE)
