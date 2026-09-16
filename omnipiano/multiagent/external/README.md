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

3. **Nothing here is imported by the training path unless it is pure torch.**
   `external/mat/*` and `external/facmac/*` hold `nn.Module`s and ARE imported at
   runtime -- exactly like MapAnything vendors VGGT and DINOv2 layers. The rest is
   test-only: HARL, PyMARL and on-policy each pin their own gym/sacred versions
   and would fight our `ray[rllib]` install.

## Why not just run HARL / PyMARL end to end?

For HAPPO we do not, and the reason is the whole point of the benchmark. Our
IPPO/MAPPO runs share one sampler, one reward, one evaluation protocol and one
metrics path. Running HAPPO inside HARL would change all four at once, so a
HAPPO-vs-MAPPO difference could no longer be attributed to the update rule. We
therefore implement the update rule in our own Learner and prove numerical
agreement with the reference (see `omnipiano/tests/test_marl_happo_oracle.py`).

For FACMAC and MASAC that trade is reversed: they need a replay buffer, target
networks and an off-policy loop that our PPO trainer does not have, and RLlib's
new API stack has open issues combining multi-agent with replay buffers. Running
them in their own framework against a shared env wrapper is the cheaper path.
The cost -- different bookkeeping -- must be stated in the paper, and the
comparison restricted to the environment-step axis.

## Provenance

| dir | upstream | commit | license | used as |
|---|---|---|---|---|
| `harl/` | PKU-MARL/HARL | see PROVENANCE.json | see UPSTREAM_LICENSE | HAPPO + HASAC oracle |
| `mat/` | PKU-MARL/Multi-Agent-Transformer | " | " | MAT module (runtime) + oracle |
| `facmac/` | oxwhirl/facmac | " | " | QMIX mixer (runtime) |
| `on_policy/` | marlbenchmark/on-policy | " | " | MAPPO retro-oracle |
