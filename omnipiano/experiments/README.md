# Unified experiment control plane

`omnipiano.experiments` is an additive control plane over the four existing
training backends. It gives Standard, Robust, Safe, and Multi-Agent RL one
versioned JSON contract without changing their trainers.

The pipeline is deliberately split into inspectable stages:

```text
schema-v3 JSON -> validate -> resolve seeds -> bind backend -> materialize -> launch
```

## Command line

Use any file in `examples/` as a starting point:

```bash
# Structural and capability validation.
python -m omnipiano.experiments validate \
  omnipiano/experiments/examples/marl.json

# Resolved defaults and one run per seed, still backend-neutral.
python -m omnipiano.experiments plan \
  omnipiano/experiments/examples/marl.json

# Exact existing-backend commands, without writes or processes.
python -m omnipiano.experiments bind \
  omnipiano/experiments/examples/marl.json

# Preview paths/artifacts, then prepare them.
python -m omnipiano.experiments materialize \
  omnipiano/experiments/examples/marl.json --dry-run
python -m omnipiano.experiments materialize \
  omnipiano/experiments/examples/marl.json

# Preview, then start detached training processes.
python -m omnipiano.experiments launch \
  omnipiano/experiments/examples/marl.json --dry-run
python -m omnipiano.experiments launch \
  omnipiano/experiments/examples/marl.json

python -m omnipiano.experiments status --root runs
python -m omnipiano.experiments stop \
  runs/EXPERIMENT/RUN/.omnipiano/process.json
```

`catalog` reports only algorithms the current backend can actually bind;
unfinished declarations are opt-in:

```bash
python -m omnipiano.experiments catalog
python -m omnipiano.experiments catalog --track marl --include-unavailable
```

## Local UI and API

Start the loopback-only API from the repository root:

```bash
python -m omnipiano.experiments serve --allowed-root runs
```

Then start the Vite UI in another terminal (see `ui/README.md`). The browser
can validate and preview freely, but asks for confirmation before launching.

## Safety properties

- all backend commands come from repository-owned adapters;
- subprocesses receive an argument vector with `shell=False`;
- generated files and runs are confined to the configured allowed root;
- materialization never overwrites content it did not create;
- every run records its command, environment, PID, and logs under
  `.omnipiano/`;
- duplicate launch attempts fail closed;
- stopping checks PID identity before sending `SIGTERM`;
- the HTTP service binds only to loopback and rejects non-loopback origins.

The canonical request schema is
`contract/experiment-v3.schema.json`. Backend-specific representability is
checked by `adapters/`, while `catalog.py` is the shared capability source of
truth. Existing backend entrypoints remain directly usable and unchanged.
