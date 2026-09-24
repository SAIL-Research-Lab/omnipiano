# Safety integration into main — 2026-09-24

This integrates safety branch `2bd8a60` with main `ce298d8`, retaining both
histories. Relative to that main commit, **every changed or added file is under
`omnipiano/safety/`**. The existing `constraints.py` is unchanged.

The root `setup.py`, README, examples, general-RL code, MARL code, robustness
code, environment factory, task implementations and wrappers are not edited.
Their latest main versions are retained rather than replaced by older safety
branch copies. Safety does not add OmniSafe to the root package's mandatory
dependencies. General-RL/MARL users do not need to install the safety backend.

## Compatibility adaptation

Current main only attaches `MidiEvaluationWrapper` and musical metrics when
`omnipiano.make(..., mode="eval")` is used. The old safety adapter used the
default training mode during checkpoint replay as well, which would omit F1.

The adaptation stays entirely in safety:

- `cmdp.py` defaults to main's `mode="train"` and supports an explicit mode.
- `evaluation_mode()` temporarily selects evaluation mode while OmniSafe's
  evaluator constructs its environment. A context-local token is restored even
  if loading raises an exception; subsequent training remains in training mode.
- `evaluate.py` uses that context only around checkpoint/environment loading.
- A regression test covers normal construction, evaluation construction,
  restoration after an exception and explicit mode selection.

Training therefore retains main's omission of expensive per-step F1 bookkeeping;
checkpoint replay still reports full-song reward, cost and F1. Safety task/cost
definitions, original paper matrices, algorithm configurations, budgets, seed
defaults and evaluation episode defaults are unchanged by this merge.

Use the updated [Quick Start](QUICKSTART.md): clone **main**, then install safety
into a **separate Python environment**. Do not combine the safety profile with
the optional MARL/TorchRL extras. The historical installation and 32-algorithm
reports remain available, clearly labelled as earlier safety-branch records.

## Validation of the merged source

Tested in a separate source directory on the RTX 4070 Ti, using the existing
isolated Python 3.10.21 / OmniSafe 0.5.0 / PyTorch 2.13.0 / NumPy 1.26.4 /
MuJoCo 3.12.0 / dm-control 1.0.45 runtime. This was a source integration test,
not a fresh installation of every current-main dependency or optional extra.

| Check | Result |
|---|---|
| `python -m pytest omnipiano/safety/tests -q` | **79 passed** |
| Selected main protocol, robust-config and MARL-entrypoint tests | **50 passed, 4 skipped** |
| All registered safety tasks: reset + 2 zero and 2 random steps | **58/58 passed**, 232 control steps |
| PPO: 2K CUDA training + initial/final checkpoint replay | Passed |
| PPOLag: 2K CUDA training + initial/final checkpoint replay | Passed |
| SACLag: 2K CUDA training + initial/final checkpoint replay | Passed |
| PPOSaute: 2K CUDA training + initial/final checkpoint replay | Passed |
| Custom reward/cost/F1 plot, PNG and PDF | Passed |

The four skipped protocol cases import SB3 TQC/baseline scripts that require
`sb3_contrib`, which is absent from the isolated safety runtime. No main test
was edited to obtain these results. This was not the entire main test suite.

Each algorithm used main task 0 (2-hand ForElise, joint-range Fraction, budget
19.95, seed 1), with one full 399-step evaluation episode at each checkpoint.
All four runs reached `status: complete`; reward/cost/F1 arrays had shape `(2, 1)`
and finite values, costs were nonnegative, and F1 lay within [0, 1]. SACLag used
the documented smoke warm-up overrides; these runs do not establish long-run
performance or validate all 32 algorithms on the newly merged main.

Tested Python-source fingerprint:

```text
541f4572f9d1a50a70cad28f6bd43a60f477c11d7aa68071ddca6e7ceea91152
```

The underlying main code has advanced since earlier safety experiments. Preserve
their original source/configuration records; do not silently pool historical
results with this merged checkout or resume them as if the code hash matched.
The previously documented Safety-Gymnasium/MuJoCo dependency-metadata exception
also remains; this merge does not claim to remove it.
