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

Bind the same plan to the existing backend CLI without launching it:

```bash
python -m omnipiano.experiments bind \
  omnipiano/experiments/examples/marl.json
```

Inspect the capabilities exposed by the current repository:

```bash
python -m omnipiano.experiments catalog
python -m omnipiano.experiments catalog --track marl --include-unavailable
```

The catalog lists only current backend algorithms as selectable; declared but unfinished
algorithms such as MAT and MASAC are visible only with `--include-unavailable`.

The backend-neutral `plan` deliberately contains:

```json
"launch": {
  "status": "unbound",
  "backend": null,
  "command": [],
  "environment": {},
  "artifacts": {}
}
```

The `bind` command fills those fields with an exact existing backend command.
It remains side-effect free: it does not write generated backend artifacts,
create run directories, import a trainer, or execute a subprocess. A MARL
custom task carries its generated schema-v2 JSON inline under
`launch.artifacts`; explicit materialization and execution belong to the next
stage. Existing Standard, Robust, Safety, and MARL entrypoints are unchanged.
The added `launch.artifacts` field makes this plan format version 2; the user
request schema remains version 3.

The canonical schema is `contract/experiment-v3.schema.json`. A task can be an
existing registered environment or a custom task description. Track-specific
capability checks live in `catalog.py`, not in the shared structural schema.
Backend-specific parameter and environment checks are enforced by the four
track adapters. Unsupported requests fail closed instead of being ignored.
