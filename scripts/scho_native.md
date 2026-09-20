# SCHO native backend

Status: implemented, external validation pending.

Baseline commit:
6a86ce3f9d3c38b07bacf522172e610cef6bffb0

IPPO/MAPPO retain their RLlib backend.
HAPPO/MAT/FACMAC/MASAC/centralized PPO use the native backend.

MASAC variant: cooperative_joint_entropy_v1.
MAT: heterogeneous adapters, fixed compiled agent order, masked action padding.
Centralized PPO preserves the original task partition and reward calculation.

Native checkpoints contain learning state and replay.
Resume resets environment episodes; exact simulator-trajectory continuation
is not claimed.

Reference commits:
HARL b1af98b0dbab72a2eee9d160751cd09aedbb8ce2
MAT be3ff49c8264d454c1fe2c41582aa2bfc98498c8
FACMAC d7e62b8c51a5a77330de85f83c10553d0bd18fe5
BenchMARL 65d649d80e0bdcbdbe2c5d6a3f02dbfed8f0bec1

Formal results must use a recorded commit after independent validation.