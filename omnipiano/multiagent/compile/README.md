# Configurable MARL experiments

The public flow has one intermediate representation and one environment
builder:

```text
experiment.json / registered env_id
              ↓
schema.py + presets.py → compiler.py → ResolvedExperiment + ResolvedTask
                                      ↓
               environment.py → env_runtime/parallel_env.py
                                      ↓
                               RLlib training
```

- `schema.py` defines the serializable contracts.
- `compiler.py` validates user JSON and expands defaults.
- `../../tasks/multi_hand_layouts.py` is the single raw source for physical hand
  layouts and key buckets. `presets.py` adds default ownership and song aliases;
  `env_runtime/topology.py` materializes runtime topology objects and calculations.
- `environment.py` snapshots physics settings, adapts old registered
  `env_id`s to `ResolvedTask`, and builds the dm_env chain.
- `env_runtime/parallel_env.py` implements PettingZoo observation/action splitting.
- `env_runtime/reachability.py` and `env_runtime/metrics.py` contain the two
  simulator-aware analyses.
- `../training/` contains shared training/evaluation bookkeeping. Algorithm
  implementations remain in `../algos/`.

There is no second legacy environment builder: `make_parallel(env_id)` first
calls `resolve_registered_task(env_id)` and then uses
`make_parallel_from_task(...)`, exactly like a schema-v2 request.

Schema v1 keeps the historical registered `env_id` workflow unchanged.
Schema v2 adds a `task` block and inherits all omitted training defaults from
`../marl_train_config.json` (or a schema-v1 file named by `extends`).

Launch the checked-in example with:

```bash
python -m omnipiano.multiagent.train omnipiano/multiagent/marl_task_example.json
```

Validate the resolved task and actual spaces without starting Ray, W&B, or a
GPU job:

```bash
python -m omnipiano.multiagent.train experiment.json --dry-run
```

The relevant schema-v2 fields are:

```json
{
  "schema_version": 2,
  "task": {
    "name": "winterwind-4h-2a-overlap",
    "song": "WinterWind",
    "num_hands": 4,
    "num_agents": 2,
    "assignment": "default",
    "sustain_owner": "secondo",
    "agents": [
      {
        "name": "secondo",
        "action_key_range": [1, 52],
        "observation_key_range": [1, 60],
        "visible_teammate_hands": "boundary"
      }
    ]
  },
  "experiment": {"algo": "ippo", "env_id": null, "seed": 0, "run_dir": null}
}
```

- Piano ranges are inclusive, user-facing **1..88** intervals. The compiler
  converts them once to internal 0..87 indices.
- `action_key_range` limits every owned hand's lateral wrist slider. It does
  not remove finger actuators and is not a hard fingertip bounding box.
- `observation_key_range` independently selects piano state and MIDI goal
  columns. Own-hand joint observations are always included.
- `visible_teammate_hands` is `"none"`, `"boundary"`, `"all"`, or an explicit
  list of zero-based hand IDs.
- `assignment` is `"default"`, `"balanced"`, or `"explicit"`. Explicit agents
  supply `hand_ids`; every hand must have exactly one owner. IDs are zero-based
  and follow the preset's physical left-to-right order.
- Action ranges may overlap. Observation ranges may differ from action ranges.

The compiler validates the request without Ray or MuJoCo. At launch,
`prepare_task` snapshots the exact physical hand and environment settings.
Training workers and deterministic evaluation then rebuild from this same
snapshot; no dynamic process-local environment registration is required.
