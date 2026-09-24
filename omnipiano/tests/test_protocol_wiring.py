"""Regression: training scripts must *read* protocol constants, not copy them.

Motivation
----------
``BenchmarkProtocolConfig`` is the single source of truth for the five
cross-library constants (``total_env_steps``, ``seed``, ``num_eval_eps``,
``gamma``, ``eval_freq_env_steps``). A script that hardcodes the *current*
value looks correct forever — its ``eval_summary.json`` reports the right
number and nothing warns — right up until the protocol constant changes,
at which point that one script silently keeps producing curves on the old
setting. This was a real defect: ``run_sb3_{sac,tqc}_template.py`` had
``--gamma`` hardcoded to ``0.8`` while the other two scripts read
``proto.gamma`` (fixed 2026-08-02).

Method
------
Equality against the current value cannot distinguish "bound" from
"coincidentally equal", so this test swaps the ``BenchmarkProtocolConfig``
symbol in each script's own module namespace for a stub carrying sentinel
values and rebuilds the parser. A bound default tracks the sentinel; a
hardcoded one does not.

Note that mutating ``BenchmarkProtocolConfig.gamma`` directly would NOT
work — dataclass bakes field defaults into the generated ``__init__``
signature at class-creation time, so class-attribute assignment never
reaches ``BenchmarkProtocolConfig()``. Hence the subclass stub.

Coverage
--------
``run_omnisafe_template.py`` is only importable inside the
``omnipiano_omnisafe`` conda env (``import omnisafe`` at module scope), so
it is skipped here and must be checked in that env. It was verified BOUND
on all four fields it exposes on 2026-08-02.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

from omnipiano.configs import BenchmarkProtocolConfig

_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples",
)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# Sentinels: deliberately unlike any plausible real default.
SENTINELS = {
    "total_env_steps": 1_234_567,
    "seed": 777,
    "num_eval_eps": 9,
    "gamma": 0.1234,
    "eval_freq_env_steps": 31_337,
}


class _StubProto(BenchmarkProtocolConfig):
    """BenchmarkProtocolConfig whose protocol fields are all sentinels."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for field, value in SENTINELS.items():
            setattr(self, field, value)


# (module, argv needed to parse at all, {protocol field: argparse dest})
SCRIPTS = [
    (
        "run_sb3_sac_template",
        [],
        {
            "total_env_steps": "total_steps",
            "seed": "seed",
            "num_eval_eps": "num_eval_eps",
            "gamma": "gamma",
            "eval_freq_env_steps": "eval_interval_env_steps",
        },
    ),
    (
        "run_sb3_tqc_template",
        [],
        {
            "total_env_steps": "total_steps",
            "seed": "seed",
            "num_eval_eps": "num_eval_eps",
            "gamma": "gamma",
            "eval_freq_env_steps": "eval_interval_env_steps",
        },
    ),
    (
        "run_sb3_baseline",
        ["--algo", "ppo"],
        {
            "total_env_steps": "total_steps",
            "seed": "seed",                  # was hardcoded 0; bound 2026-08-02
            "num_eval_eps": "num_eval_eps",  # was hardcoded 1; bound 2026-08-02
            "gamma": "gamma",
            "eval_freq_env_steps": "eval_freq",
        },
    ),
    (
        "omnipiano.multiagent.train",
        [],
        {
            "total_env_steps": "total_steps",
            "seed": "seed",
            "num_eval_eps": "num_eval_eps",
            "gamma": "gamma",
            "eval_freq_env_steps": "eval_freq",
        },
    ),
]


def _load(modname):
    try:
        return importlib.import_module(modname)
    except ImportError as e:  # missing sb3 / sb3_contrib in this env
        pytest.skip(f"{modname} not importable here: {e}")


def _build_parser(mod):
    """Support canonical modules and legacy modules during migration."""
    builder = getattr(mod, "build_arg_parser", None)
    if builder is None:
        builder = getattr(mod, "_build_arg_parser")
    return builder()


@pytest.mark.parametrize("modname,argv,mapping", SCRIPTS,
                         ids=[s[0] for s in SCRIPTS])
def test_protocol_defaults_are_bound(modname, argv, mapping):
    """Each mapped argparse default must track BenchmarkProtocolConfig."""
    mod = _load(modname)
    real = BenchmarkProtocolConfig()

    # 1. Unpatched: defaults equal the real protocol values.
    args = _build_parser(mod).parse_args(argv)
    for field, dest in mapping.items():
        assert getattr(args, dest) == getattr(real, field), (
            f"{modname}: --{dest.replace('_', '-')} default "
            f"{getattr(args, dest)!r} != protocol {field}="
            f"{getattr(real, field)!r}"
        )

    # 2. Patched: defaults follow the sentinel => genuinely bound, not
    #    coincidentally equal.
    mod.BenchmarkProtocolConfig = _StubProto
    try:
        args = _build_parser(mod).parse_args(argv)
    finally:
        mod.BenchmarkProtocolConfig = BenchmarkProtocolConfig
    for field, dest in mapping.items():
        assert getattr(args, dest) == SENTINELS[field], (
            f"{modname}: --{dest.replace('_', '-')} is HARDCODED "
            f"({getattr(args, dest)!r}) instead of reading "
            f"BenchmarkProtocolConfig.{field}"
        )


def test_periodic_eval_honours_num_eval_eps():
    """``--num-eval-eps`` must reach EvalCallback, not just the final eval.

    Both templates pinned ``n_eval_episodes=1`` at the EvalCallback call site
    while accepting ``--num-eval-eps`` on the CLI, so the flag silently did
    nothing to the learning curve (fixed 2026-08-28). Grepping the call site is
    crude but it is the one thing that actually distinguishes the two states --
    parsing args cannot see a constant baked in fifty lines later."""
    import re
    for name in ("run_sb3_sac_template", "run_sb3_tqc_template", "run_sb3_baseline"):
        src = open(os.path.join(_EXAMPLES, name + ".py")).read()
        calls = re.findall(r"n_eval_episodes\s*=\s*([^\s,)]+)", src)
        assert calls, "%s: no EvalCallback n_eval_episodes found" % name
        for c in calls:
            assert "num_eval_eps" in c, (
                "%s: n_eval_episodes=%s is hardcoded; it must read "
                "args.num_eval_eps so the periodic eval honours the protocol"
                % (name, c))


def test_default_seed_is_a_member_of_the_replication_set():
    """``seed`` is the single-run default; ``seeds`` is what a reported
    number must cover. Omitting ``--seed`` must therefore yield replicate
    #1 of the reported set, not an orphan run outside it — this is exactly
    what went wrong with the inherited default of 42, which sits outside
    ``(0, 1, 2)``. Enforced here rather than in ``__post_init__`` so that
    one-off ``BenchmarkProtocolConfig(seed=...)`` stays legal."""
    proto = BenchmarkProtocolConfig()
    assert len(proto.seeds) >= 3, (
        f"replication set {proto.seeds} is below the 3-seed floor"
    )
    assert len(set(proto.seeds)) == len(proto.seeds), (
        f"replication set {proto.seeds} contains duplicates"
    )
    assert proto.seed == proto.seeds[0], (
        f"default seed {proto.seed} != seeds[0]={proto.seeds[0]}; running "
        "without --seed would produce a run outside the reported set"
    )


@pytest.mark.parametrize("modname,argv,mapping", SCRIPTS,
                         ids=[s[0] for s in SCRIPTS])
def test_explicit_cli_still_overrides_protocol(modname, argv, mapping):
    """Binding must not make the protocol value unoverridable — ablations
    (e.g. ``--gamma 0.99`` to reproduce library defaults) depend on this."""
    mod = _load(modname)
    args = _build_parser(mod).parse_args(argv + ["--gamma", "0.99"])
    assert args.gamma == 0.99
