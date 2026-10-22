"""Side-effect-free binding from unified plans to existing backend CLIs."""

from __future__ import annotations

from typing import Callable

from omnipiano.experiments.adapters.common import AdapterError, replace_runs
from omnipiano.experiments.models import PlanBundle, ResolvedExperiment, RunPlan


Adapter = Callable[[ResolvedExperiment, RunPlan], RunPlan]


def _adapter(track: str) -> Adapter:
    if track == "standard":
        from omnipiano.experiments.adapters.standard import bind_run
    elif track == "robust":
        from omnipiano.experiments.adapters.robust import bind_run
    elif track == "safety":
        from omnipiano.experiments.adapters.safety import bind_run
    elif track == "marl":
        from omnipiano.experiments.adapters.marl import bind_run
    else:  # The core schema/catalog should make this unreachable.
        raise AdapterError(f"no backend adapter for track {track!r}")
    return bind_run


def bind_plan(bundle: PlanBundle) -> PlanBundle:
    """Return a new bundle whose runs are bound, without writing or executing."""
    bind_run = _adapter(bundle.resolved_experiment.track)
    runs = [bind_run(bundle.resolved_experiment, run) for run in bundle.runs]
    return replace_runs(bundle, runs)


__all__ = ["AdapterError", "bind_plan"]
