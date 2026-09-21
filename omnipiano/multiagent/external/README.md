# Vendored reference implementations

Three rules. They exist so that "we match the reference implementation" is a
claim a reviewer can check in one command instead of taking on trust.

1. **Files copied from upstream are VERBATIM and are never edited.**
   Not reformatted, not linted, not type-annotated. The MARL provenance test records
   a sha256 for each and fails if any changes. That is what makes
   "unmodified reference implementation" checkable rather than asserted.

2. **Every adaptation lives in `<name>/_adapter.py`, which we wrote.**
   Reference code is a class method on a framework object; our tests need a pure
   function. The adapter does that conversion and NOTHING else -- no fixed bugs,
   no changed constants. If the adapter has to make a choice (a default the
   upstream config supplied), it documents the choice in a comment.

3. **Nothing here is imported by the training path unless it is pure torch and
   redistributable.** The rest is test-only: HARL, PyMARL and on-policy each pin
   their own gym/sacred versions and would fight our `ray[rllib]` install.

## Why not just run HARL / PyMARL end to end?

For HAPPO we do not, and the reason is the whole point of the benchmark. Our
IPPO/MAPPO runs share one sampler, one reward, one evaluation protocol and one
metrics path. Running HAPPO inside HARL would change all four at once, so a
HAPPO-vs-MAPPO difference could no longer be attributed to the update rule. We
therefore implement the update rule in our own Learner and prove numerical
agreement with the reference (see `omnipiano/tests/test_marl_happo_oracle.py`).

FACMAC uses OmniPiano's native joint-transition backend: it keeps the same task,
sampler, evaluation, metrics and checkpoint lifecycle as the other baselines,
but supplies the replay buffer, target networks and off-policy update that PPO
does not have. The upstream oxwhirl repository has no licence, so its code is
not copied here. Our clean-room implementation pins the read-only reference
commit in every configuration and has architecture/equation/gradient tests.

## Provenance

| dir | upstream | commit | license | used as |
|---|---|---|---|---|
| `harl/` | PKU-MARL/HARL | see PROVENANCE.json | see UPSTREAM_LICENSE | HAPPO + HASAC oracle |
| `mat/` | PKU-MARL/Multi-Agent-Transformer | " | " | MAT module (runtime) + oracle |
| FACMAC (not vendored) | oxwhirl/facmac | `d7e62b8...` | no licence found | read-only specification/oracle |
| `on_policy/` | marlbenchmark/on-policy | " | " | MAPPO retro-oracle |
