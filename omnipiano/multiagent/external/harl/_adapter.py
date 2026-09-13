"""Adapt HARL's HAPPO update into a pure function our tests can call.

WE WROTE THIS FILE. The vendored `happo.py` next to it is verbatim upstream and
is never edited (see ../README.md, rule 1 and 2).

What an adapter may do:      reshape arguments, supply a default the upstream
                             config would have supplied, extract the algebra
                             from a method into a function.
What an adapter may NOT do:  change a constant, fix a bug, reorder the maths.
Every choice forced on us is recorded in a comment, because an undocumented
choice here would silently make the oracle test compare the wrong thing.
"""

from __future__ import annotations

import json
import pathlib

import torch

_PROV = json.loads(
    (pathlib.Path(__file__).with_name("PROVENANCE.json")).read_text(encoding="utf-8"))
UPSTREAM_URL: str = _PROV["upstream_url"]
UPSTREAM_COMMIT: str = _PROV["upstream_commit"]


def happo_surrogate_reference(
    *,
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    compound_log_factor: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """HARL's HAPPO surrogate, called through as thin a shim as possible.

    Returns the quantity to be MAXIMISED, matching our
    ``_happo_math.happo_surrogate`` convention. If upstream returns a LOSS
    (negated) or a MEAN (reduced), this shim un-negates / un-reduces and says so
    below -- an unnoticed sign or reduction difference would make the oracle test
    pass on the wrong quantity, which is worse than no oracle test at all.
    """
    raise NotImplementedError(
        "Fill this in from the vendored happo.py. Run:\n"
        "  bash scripts/fetch_external.sh locate harl\n"
        "  python scripts/dump_external_source.py harl happo\n"
        "then paste the printed source. Three things must be pinned down:\n"
        "  (1) does upstream take the compound factor, or its log?\n"
        "  (2) does it return a loss (negated) or a surrogate?\n"
        "  (3) does it reduce (mean) internally, and over which axis?\n"
        "Guessing any of these makes the oracle compare the wrong quantity.")