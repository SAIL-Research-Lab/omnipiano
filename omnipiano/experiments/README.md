# Unified experiment control plane

`omnipiano.experiments` is an additive, backend-neutral layer. It validates a
single versioned JSON format and expands its seed list into deterministic run
plans. It does not import a training backend or launch a process.

Validate an example:

```bash
python -m omnipiano.experiments validate \
  omnipiano/experiments/examples/marl.json
```

Preview resolved defaults and seed fan-out:

```bash
python -m omnipiano.experiments plan \
  omnipiano/experiments/examples/marl.json
```

Inspect the capabilities exposed by the current repository:

```bash
python -m omnipiano.experiments catalog
python -m omnipiano.experiments catalog --track marl --include-unavailable
```

The catalog distinguishes backend support from unified-adapter readiness. It
lists only current backend algorithms as selectable; declared but unfinished
algorithms such as MAT and MASAC are visible only with `--include-unavailable`.

The phase-two plan deliberately contains:

```json
"launch": {
  "status": "unbound",
  "backend": null,
  "command": [],
  "environment": {}
}
```

Backend adapters will bind these fields in a later phase. Until then, this
module cannot start training and therefore cannot alter the behavior of the
existing Standard, Robust, Safety, or MARL entrypoints.

The canonical schema is `contract/experiment-v3.schema.json`. A task can be an
existing registered environment or a custom task description. Track-specific
capability checks live in `catalog.py`, not in the shared structural schema.
Backend-specific parameter and environment checks remain the responsibility of
the adapters added in the next phase.
