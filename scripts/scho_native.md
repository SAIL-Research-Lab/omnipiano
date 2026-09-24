# SCHO native backend

Status: FACMAC supported; the other native algorithms retain their individual
registry gates (experimental/planned) until their own validation is complete.

Baseline commit:
6a86ce3f9d3c38b07bacf522172e610cef6bffb0

IPPO/MAPPO retain their RLlib backend.
FACMAC uses the native backend. Other algorithms must follow the backend named
by their current `AlgoSpec`; this document does not promote them implicitly.

FACMAC readiness evidence:

- continuous deterministic per-agent actors and per-agent utilities use the
  reference 400x400 ReLU architecture with tanh action output;
- the clean-room QMIX mixer matches the pinned upstream tensor equation and is
  tested for monotonic gradients;
- replay, 10k random exploration, 1k learner warm-up, UTD=1, Adam settings,
  action regularization and soft targets follow `facmac_mamujoco.yaml`;
- all actors receive the current joint-action policy gradient;
- true termination/truncation, replay sampling, target update, checkpoint
  resume and deterministic evaluation are tested;
- parameter sharing is disabled intentionally because SCHO supports unequal
  observation/action dimensions. Gamma 0.8 and 10M steps remain task protocol
  choices rather than FACMAC defaults.

MASAC variant: cooperative_joint_entropy_v1.
MAT: heterogeneous adapters, fixed compiled agent order, masked action padding.
Centralized PPO preserves the original task partition and reward calculation.

Native periodic/final policy checkpoints contain model, targets, optimizers and
RNG state but omit replay.  Off-policy runs additionally maintain one rotating
recovery checkpoint containing the newest configured replay tail (FACMAC:
50,000 transitions); older recovery snapshots are removed after the new one is
complete.  This preserves the official 1M training replay without multiplying
it across every historical checkpoint.  Resume resets environment episodes;
exact simulator-trajectory continuation is not claimed.

Reference commits:
HARL b1af98b0dbab72a2eee9d160751cd09aedbb8ce2
MAT be3ff49c8264d454c1fe2c41582aa2bfc98498c8
FACMAC d7e62b8c51a5a77330de85f83c10553d0bd18fe5
BenchMARL 65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1

Formal results must use a recorded commit after independent validation.
