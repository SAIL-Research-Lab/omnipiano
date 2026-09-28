# MARL reproduction tools

This package contains the operational tools behind the frozen WinterWind
SCHO 5 x 5 x 3 suite:

- `generate_scho_suite.py` creates complete task/run configurations;
- `verify_reproduction.py` validates provenance, hashes, and the 75 cells;
- `audit_tasks.py` constructs and probes the five environments;
- `run_queue.py` assigns JSON runs to isolated GPU slots.

These modules are not part of the algorithm or environment runtime. A single
configuration still runs directly through `python -m omnipiano.multiagent.train`.
Use the suite-level `configs/scho_winterwind_5x5x3/reproduce.sh` entry point
instead of calling the modules separately for a paper reproduction.
