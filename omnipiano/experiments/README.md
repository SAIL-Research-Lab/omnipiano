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
capability checks intentionally belong to the future catalog and adapters,
not to the shared schema.
