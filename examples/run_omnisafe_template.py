"""OmniSafe (PPOLag by default) template for OmniPiano.

Status: production entry point for the OmniSafe baseline runs (PPOLag).
The structural pieces — CMDP adapter, ``@env_register`` hookup,
``BenchmarkProtocolConfig`` wiring, seeded final eval,
``eval_summary.json`` dump — are all in place and CLI-driven
(parallel to ``examples/run_sb3_*_template.py``). Running this
template requires ``pip install omnisafe``; the dependency is
intentionally NOT declared in OmniPiano's ``setup.py`` because OmniSafe
is one optional backend among several.

Why this template
-----------------
Demonstrates how an OmniSafe user plugs into OmniPiano:

  1. Wrap any registered OmniPiano gymnasium env once as a CMDP
     subclass. The class-level ``_support_envs`` is populated
     dynamically from ``omnipiano.envs.registration._registry`` so all
     24+ registered task ids are available without manual listing.
     OmniSafe drives rollouts itself through
     ``omnisafe.Agent(...).learn()`` — we do not write a rollout loop.
  2. Read **algorithm-agnostic** protocol constants (env-step budget,
     scalar seed, num_eval_eps) from ``BenchmarkProtocolConfig`` and
     pass them into OmniSafe via ``custom_cfgs``. Algorithm-specific
     hyperparameters (``cost_limit``, ``lagrangian_multiplier_init``,
     ``lambda_lr``, ...) come from CLI flags with paper PPOLag
     defaults — same pattern as the SB3 templates.
  3. Final benchmark eval uses ``omnisafe.Evaluator`` only for
     ``load_saved`` (env + actor reconstruction); the rollout itself is
     a bespoke seeded loop in ``_final_eval`` that also captures
     terminal info dicts — see ``Evaluator caveats`` below and
     ``_final_eval``'s docstring for why ``Evaluator.evaluate`` is
     bypassed.

CLI
---
::

    python examples/run_omnisafe_template.py \\
        --env OmniPiano-ForElise-WristLimit-v0 \\
        --experiment-name ppolag_forelise_wrist \\
        --total-steps 5000000 \\
        --seed 42 \\
        --vector-env-nums 1 \\
        --cost-limit 25.0

Smoke-test by adding ``--smoke-test`` (drops total_steps to 20K and
num_eval_eps to 1).

Default env is a safety task (``OmniPiano-ForElise-WristLimit-v0``)
because PPOLag is a *constrained* RL algorithm — running it on a task
with no safety constraints would degenerate into vanilla PPO with a
trivial Lagrangian multiplier. Switch to any registered task with
``--env``.

Seeding
-------
OmniSafe accepts a single scalar seed via ``custom_cfgs={'seed': ...}``
(default in ``PPOLag.yaml`` is ``seed: 0``; we override). Evaluator has
no seed parameter — eval RNG comes from whatever env state remains after
``load_saved``. For paper-style replication, run this script N times with
distinct seeds and aggregate ``eval_summary.json`` files offline (same as
the SB3 templates and the RoboPianist paper).

OmniPiano-side glue
-------------------
* Per-step cost is in ``info[InfoKeys.STEP_SAFETY_COST_TOTAL]`` (the gym
  side returns the standard 5-tuple; cost lives in ``info``). The CMDP
  adapter extracts it and returns it as the third element of OmniSafe's
  6-tuple.
* Observations are already flat ``Box(shape=(N,), dtype=float32)`` after
  the dm_env-level ``ConcatObservationWrapper`` + ``DmEnvToGymnasium``
  adapter. No ``gym.wrappers.FlattenObservation`` is needed (older
  versions of this template wrapped it; the wrap is now redundant given
  OmniPiano's paper-chain layout).

Evaluator caveats
-----------------
``omnisafe.Evaluator.evaluate()`` returns only ``(rewards, costs)`` —
it neither seeds its resets nor surfaces terminal ``info`` dicts. That
is exactly why ``_final_eval`` drives its own seeded rollout loop after
``load_saved``: it captures terminal infos (``episode_task/f1`` and
friends) and summarizes every scalar key into ``eval_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, ClassVar, Dict, List

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch

import omnisafe
from omnisafe.envs.core import CMDP, env_register

from omnipiano import make as omnipiano_make
from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.envs.registration import _registry as _omnipiano_registry
from omnipiano.utils.info_keys import InfoKeys


# OmniSafe groups constrained algorithms into two config schemas — verified by
# grepping ``omnisafe/configs/**/*.yaml`` for ``lagrange_cfgs:`` vs
# ``algo_cfgs.cost_limit:``. ``custom_cfgs`` must route ``cost_limit`` to the
# right section per algorithm or ``recursive_check_config`` raises
# ``KeyError: Invalid key``. Algos not in either set (PPO / SAC / TRPO / DDPG /
# TD3, plus the Saute / Simmer state-augment variants) take no constraint
# hparam — running them on a safety task degenerates to vanilla unconstrained
# RL; the user gets a warning at construction time.
_LAGRANGIAN_ALGOS = frozenset({
    "PPOLag", "TRPOLag", "SACLag", "DDPGLag", "TD3Lag",
    "DDPGPID", "TD3PID", "SACPID", "CPPOPID", "TRPOPID",
    "FOCOPS", "CUP", "IPO", "RCPO", "PDO",
    "BCQLag", "CCRR", "CAPPETS",
})
_TRUST_REGION_CONSTRAINED_ALGOS = frozenset({
    "CPO", "PCPO", "P3O", "OnCRPO",
    "PPOEarlyTerminated", "TRPOEarlyTerminated",
    "CCEPETS", "RCEPETS", "SafeLOOP", "COptiDICE",
})

# Off-policy algorithms inherit from DDPG (replay buffer + target nets) and
# share the off-policy adapter's training loop (one ``update_iters`` round
# of grad steps per ``update_cycle`` env-step collection — see
# ``omnisafe/algorithms/off_policy/ddpg.py:382``). For these algorithms:
#   * ``--update-iters N`` is meaningful (routed to ``algo_cfgs.update_iters``)
#   * default ``steps_per_epoch`` is 2000 in their yaml (not 20_000 like on-policy)
#   * UTD (update-to-data ratio) = update_iters / (update_cycle × vector_env_nums)
#     SB3 SAC default is UTD=1 (n_envs=1, gradient_steps=1, train_freq=1).
#     SB3 SAC paper-matched is n_envs=24, gradient_steps=24, train_freq=1 → UTD=1.
#     OmniSafe equivalent: --vector-env-nums 24 --update-iters 24 (update_cycle=1
#     in yaml default).
_OFF_POLICY_ALGOS = frozenset({
    "DDPG", "TD3", "SAC",
    "DDPGLag", "TD3Lag", "SACLag",
    "DDPGPID", "TD3PID", "SACPID",
})


@env_register
class OmniPianoCMDP(CMDP):
    """Thin CMDP adapter wrapping any registered OmniPiano gymnasium env.

    Responsibilities:
      * Convert numpy <-> torch across step/reset boundaries.
      * Pull per-step cost out of ``info[InfoKeys.STEP_SAFETY_COST_TOTAL]``
        and return it as the CMDP cost tensor (third element of the
        6-tuple OmniSafe expects).

    Note: observations are NOT flattened here — they reach the gym layer
    already as flat ``Box`` (the dm_env chain runs ``ConcatObservationWrapper``
    before the ``DmEnvToGymnasium`` boundary). Older versions of this
    adapter wrapped ``gym.wrappers.FlattenObservation`` defensively;
    that's redundant under the current paper-chain layout.
    """

    # Populated dynamically from omnipiano's task registry — every
    # registered ``OmniPiano-*-v0`` id becomes available to OmniSafe
    # without manual listing here. ``omnipiano.envs.registration._registry``
    # is fully populated by the time this module is imported, because
    # ``from omnipiano import make`` triggers ``omnipiano.envs.__init__``
    # (which runs all the ``register(...)`` calls) before we reach this
    # class definition.
    _support_envs: ClassVar[List[str]] = sorted(_omnipiano_registry.keys())

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False  # OmniPiano episode length is MIDI-derived

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        self._num_envs = 1
        self._device = kwargs.get("device", torch.device("cpu"))
        # OmniSafe constructs the env per rollout worker; we just delegate
        # the heavy lifting (registry lookup, dm_env chain, gym wrappers)
        # to ``omnipiano.make``.
        self._env = omnipiano_make(env_id)
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]
    ]:
        act_np = action.detach().cpu().numpy().astype(np.float64)
        obs, reward, terminated, truncated, info = self._env.step(act_np)
        cost = float(info.get(InfoKeys.STEP_SAFETY_COST_TOTAL, 0.0))
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            torch.as_tensor(reward, dtype=torch.float32, device=self._device),
            torch.as_tensor(cost, dtype=torch.float32, device=self._device),
            torch.as_tensor(bool(terminated), dtype=torch.bool, device=self._device),
            torch.as_tensor(bool(truncated), dtype=torch.bool, device=self._device),
            info,
        )

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            info,
        )

    def set_seed(self, seed: int) -> None:
        # gymnasium envs accept seed at reset() time; trigger a reset.
        # OmniPiano's adapter rebuilds the underlying dm_env chain on
        # any non-None seed (see ``omnipiano/envs/dm_env_adapter.py``).
        self._env.reset(seed=seed)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI surface — mirrors ``run_sb3_sac_template.py`` for cross-template parity."""
    proto = BenchmarkProtocolConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ---- experiment identity ----
    p.add_argument(
        "--env",
        default="OmniPiano-ForElise-WristLimit-v0",
        help="Registered OmniPiano task id. Default is a safety task so "
        "PPOLag has a non-trivial cost signal; see module docstring.",
    )
    p.add_argument(
        "--experiment-name",
        default="omnisafe_ppolag",
        help="Subdir name under examples/logs/. A run_id suffix is appended.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run for sanity check: total_steps=20000, num_eval_eps=1.",
    )

    # ---- algo-agnostic protocol (BenchmarkProtocolConfig defaults) ----
    p.add_argument("--total-steps", type=int, default=proto.total_env_steps)
    p.add_argument("--seed", type=int, default=proto.seed)
    p.add_argument("--num-eval-eps", type=int, default=proto.num_eval_eps,
                   help="Final benchmark eval episode count.")

    # ---- OmniSafe vec env ----
    p.add_argument("--vector-env-nums", type=int, default=1,
                   help="OmniSafe vec env count (analogous to SB3 --n-envs). "
                   "Each value spawns N subprocess workers each running one "
                   "OmniPianoCMDP. n=8 on a 16-core machine is the sweet spot — "
                   "leave the upper 16 hyper-threads for OS + IO. Pair with "
                   "--torch-threads 2 to keep total thread count = physical cores.")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="Per-process torch CPU thread count. OmniSafe yaml "
                   "default = 16 — when combined with --vector-env-nums > 1 "
                   "this over-subscribes the CPU (8 procs × 16 threads = 128 "
                   "threads on a 32-thread machine). Set to ceil(physical_cores "
                   "/ vector_env_nums); e.g., 16 cores ÷ 8 procs = 2 threads each. "
                   "Omit to inherit OmniSafe's 16. NOTE: when --parallel > 1, "
                   "torchrun auto-sets OMP_NUM_THREADS=MKL_NUM_THREADS=1 per "
                   "subprocess, so this flag is largely irrelevant in that mode.")
    p.add_argument("--parallel", type=int, default=1,
                   help="OmniSafe asynchronous agent parallelism (A3C-style). "
                   "OmniSafe internally invokes torchrun --nproc_per_node=N to "
                   "spawn N agent subprocesses; gradients sync per epoch via "
                   "PyTorch distributed (gloo on CPU, nccl on GPU). No external "
                   "MPI install needed — torchrun ships with PyTorch. "
                   "ON-POLICY ONLY (PPO/PPOLag/CPO/FOCOPS/TRPO-family); "
                   "OmniSafe asserts parallel==1 for off-policy/model-based/"
                   "offline algos. n=8 sweet spot on 16-core CPU.")
    p.add_argument("--save-model-freq", type=int, default=None,
                   help="OmniSafe ckpt save frequency in EPOCH units. One "
                   "epoch is always steps_per_epoch env-steps (20k on-policy, "
                   "2k off-policy) — neither --parallel nor --vector-env-nums "
                   "changes that, they only redistribute those steps. If "
                   "omitted, auto-derived toward proto.eval_freq_env_steps, "
                   "which is NOT exactly reachable (50k is not a multiple of "
                   "20k, so the derivation lands on 60k). Prefer "
                   "``--save-model-freq 1`` for the densest 20k grid — it can "
                   "be subsampled later; a coarse grid cannot be refined.")
    # OmniSafe's algo_wrapper does torch.cuda.set_device(device) which requires
    # an explicit index ('cuda:0'), not bare 'cuda'. Pass 'cuda:0' / 'cuda:1' /
    # 'cpu' through verbatim — validation is left to torch at construction time.
    p.add_argument("--device", default="cpu",
                   help="torch device. Pass 'cpu', 'cuda:0', 'cuda:1', etc. "
                   "Bare 'cuda' is rejected by omnisafe's algo_wrapper.")

    # ---- algorithm choice + paper PPOLag hparams ----
    p.add_argument("--algorithm", default="PPOLag",
                   help="OmniSafe algorithm class name (PPOLag, CPO, SACLag, ...).")
    p.add_argument("--cost-limit", type=float, default=25.0,
                   help="Per-episode constraint budget. Paper PPOLag default 25.0. "
                   "Routed to ``lagrange_cfgs.cost_limit`` for Lagrangian algos "
                   "(PPOLag / TRPOLag / SACLag / FOCOPS / CUP / ...) or "
                   "``algo_cfgs.cost_limit`` for trust-region algos "
                   "(CPO / PCPO / P3O / OnCRPO / ...). Ignored for vanilla "
                   "unconstrained algos (PPO / SAC / TRPO / ...).")
    p.add_argument("--lagrangian-multiplier-init", type=float, default=None,
                   help="Initial Lagrangian multiplier. If omitted, inherits "
                   "OmniSafe yaml default (PPOLag = 0.001 — already paper value). "
                   "Pass a float ONLY for ablation. LAGRANGIAN-FAMILY ONLY "
                   "(PPOLag / TRPOLag / ...); silently ignored for CPO / PCPO / "
                   "vanilla algos.")
    p.add_argument("--lambda-lr", type=float, default=None,
                   help="Lagrangian multiplier learning rate. If omitted, inherits "
                   "OmniSafe yaml default (PPOLag = 0.035 — already paper value). "
                   "Pass a float ONLY for ablation. LAGRANGIAN-FAMILY ONLY; "
                   "ignored otherwise.")
    p.add_argument("--update-iters", type=int, default=None,
                   help="OFF-POLICY ONLY: grad steps per ``_update`` call (one "
                   "call per ``update_cycle`` env-step collection; see "
                   "``omnisafe/algorithms/off_policy/ddpg.py:382``). UTD = "
                   "update_iters / (update_cycle × vector_env_nums). SB3 SAC "
                   "convention is UTD=1: paired with ``--vector-env-nums N``, "
                   "pass ``--update-iters N`` to keep UTD=1 (otherwise UTD "
                   "drops to 1/N and SAC barely learns). Ignored for on-policy "
                   "algorithms (PPO / PPOLag / TRPO / CPO / ...).")
    p.add_argument("--buffer-size", type=int, default=None,
                   help="OFF-POLICY ONLY: replay buffer capacity (per env). "
                   "OmniSafe yaml default = 1_000_000. CAUTION: OmniSafe "
                   "VectorOffPolicyBuffer allocates buffer_size × num_envs × "
                   "(obs + next_obs + ...) slots — NOT shared like SB3 — so "
                   "with OmniPiano obs_dim=1176 the memory cost is roughly "
                   "9.6 GB × num_envs × (buffer_size / 1_000_000). For "
                   "--vector-env-nums 8 keep --buffer-size ≤ ~200000 to fit "
                   "in 16 GB. Ignored for on-policy algorithms.")

    # ---- task-property hparam: gamma (unified across SAC/PPO/PPOLag/...) ----
    p.add_argument(
        "--gamma", type=float, default=proto.gamma,
        help=f"Reward discount factor. OmniPiano protocol default = {proto.gamma} "
        "(unified across all algos as a task property; see "
        "BenchmarkProtocolConfig.gamma). Pass --gamma 0.99 to reproduce "
        "OmniSafe library default behavior for ablation.",
    )
    p.add_argument(
        "--cost-gamma", type=float, default=None,
        help="Cost discount factor. Defaults to --gamma value (cost has the same "
        "temporal structure as reward in piano tasks: hand collisions / wrist "
        "limits manifest NOW, not 100 steps later). Pass an explicit value to "
        "decouple from --gamma.",
    )

    return p


def _final_eval(save_dir: str, num_eval_eps: int, train_seed: int) -> Dict[str, Any]:
    """Run final benchmark eval on the latest checkpoint.

    We use ``omnisafe.Evaluator`` only for ``load_saved`` (env+actor
    construction, wrapper stack, ckpt loading) and then drive our own
    rollout loop instead of calling ``Evaluator.evaluate``. Two reasons:

      1. ``Evaluator.evaluate`` calls ``self._env.reset()`` with no seed
         (omnisafe/evaluator.py:425-426), so the eval RNG depends on
         whatever state the env happens to be in after ``load_saved``.
         We pass ``seed = train_seed + protocol.eval_seed_offset + ep_i``
         per episode
         (matches SB3 ``run_baseline.py:_final_eval`` convention) so the
         eval is reproducible without coupling to training RNG state.

      2. ``Evaluator.evaluate`` returns only ``(rewards, costs)`` and
         drops the terminal ``info`` dict, so RoboPianist piano metrics
         (``episode_task/f1`` / ``episode_task/key_precision`` / ...)
         never surface. Our loop captures them.

    We do not modify upstream OmniSafe (READ-ONLY clone at
    ``/home/accelerator/SafeRoboPianist/omnisafe``); bypassing
    ``Evaluator.evaluate`` keeps that constraint while delivering both
    fixes in ~15 lines of code.
    """
    # render_mode default 'rgb_array' avoids omnisafe's __set_render_mode
    # NotImplementedError on None. We don't actually render frames.
    evaluator = omnisafe.Evaluator()
    torch_save_dir = os.path.join(save_dir, "torch_save")
    # OmniSafe writes ``epoch-N.pt`` where N is unpadded. Sort numerically —
    # lexicographic ``sorted()`` would place ``epoch-900.pt`` after
    # ``epoch-5000.pt`` and silently load an earlier checkpoint than the
    # final one. Bites only for runs with > 1000 saved epochs but is free
    # to fix.
    def _epoch_num(fname: str) -> int:
        m = re.match(r"epoch-(\d+)\.pt$", fname)
        return int(m.group(1)) if m else -1
    pt_files = sorted(
        (f for f in os.listdir(torch_save_dir) if f.endswith(".pt")),
        key=_epoch_num,
    )
    if not pt_files:
        raise RuntimeError(f"No .pt checkpoints found under {torch_save_dir}")
    evaluator.load_saved(save_dir=save_dir, model_name=pt_files[-1])
    env = evaluator._env
    actor = evaluator._actor
    if env is None or actor is None:
        raise RuntimeError(f"Evaluator.load_saved did not initialize env+actor for {pt_files[-1]}")

    # Evaluation uses the same algorithm-independent seed stream as SB3:
    # train_seed + protocol.eval_seed_offset + ep_i. This decouples eval RNG from
    # training RNG state; per-ep offset gives distinct realizations even
    # though OmniPiano's default (`_randomize_hand_positions=False`)
    # makes init pose deterministic.
    eval_seed_offset = BenchmarkProtocolConfig().eval_seed_offset

    returns: List[float] = []
    costs_: List[float] = []
    terminal_infos: List[Dict[str, Any]] = []
    for ep_i in range(num_eval_eps):
        obs, _ = env.reset(seed=train_seed + eval_seed_offset + ep_i)
        ep_ret, ep_cost = 0.0, 0.0
        terminal_info: Dict[str, Any] = {}
        done = False
        while not done:
            with torch.no_grad():
                act = actor.predict(obs, deterministic=True)
            obs, rew, cost, term, trunc, info = env.step(act)
            ep_ret += float(rew.item() if hasattr(rew, "item") else rew)
            ep_cost += float(cost.item() if hasattr(cost, "item") else cost)
            done = bool(
                (term.item() if hasattr(term, "item") else term)
                or (trunc.item() if hasattr(trunc, "item") else trunc)
            )
            if done:
                terminal_info = info if isinstance(info, dict) else {}
        returns.append(ep_ret)
        costs_.append(ep_cost)
        terminal_infos.append(terminal_info)

    # Aggregate scalar piano / safety metrics across episodes. Only keys
    # whose values are numeric scalars are summarized — ``original_obs``
    # (an ndarray) and similar are skipped.
    def _scalar_keys(infos: List[Dict[str, Any]]) -> List[str]:
        keys: set[str] = set()
        for info in infos:
            for k, v in info.items():
                if isinstance(v, (bool, int, float, np.integer, np.floating)):
                    keys.add(k)
        return sorted(keys)

    summary = {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "cost_mean": float(np.mean(costs_)) if costs_ else 0.0,
        "cost_std": float(np.std(costs_)) if costs_ else 0.0,
    }
    for k in _scalar_keys(terminal_infos):
        vals = [float(info[k]) for info in terminal_infos if k in info]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

    return {"returns": returns, "costs": costs_, "summary": summary}


def main():
    args = _build_arg_parser().parse_args()

    if args.smoke_test:
        args.total_steps = 20_000
        args.num_eval_eps = 1

    proto = BenchmarkProtocolConfig()

    # Resolve experiment dir under examples/logs/<experiment-name>_<run_id>/
    # — same pattern the SB3 templates use, just hand-rolled to avoid
    # depending on SB3's ``get_latest_run_id``.
    #
    # Distributed-launch caveat: when ``--parallel > 1``, OmniSafe internally
    # calls ``torchrun --nproc_per_node=N`` to re-launch THIS script in N
    # subprocesses (each re-executes ``main()`` from the top). Without
    # coordination, each child re-resolves run_id and creates its own
    # ``_<N+1>`` dir — five parallel ranks would produce five log_dirs,
    # ranks would write to different paths, and rank-0's progress.csv would
    # disagree with each child's own ckpt save_dir.
    #
    # Fix: parent process exports ``OMNIPIANO_LOG_DIR`` env var; children
    # detect torchrun's ``MASTER_ADDR`` env var and re-use parent's log_dir
    # instead of re-resolving. ``subprocess.check_call(args, env=...)`` in
    # ``omnisafe/utils/distributed.py:fork()`` does ``env = os.environ.copy()``
    # so OMNIPIANO_LOG_DIR propagates automatically.
    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_root, exist_ok=True)

    parent_log_dir = os.environ.get("OMNIPIANO_LOG_DIR")
    if "MASTER_ADDR" in os.environ and parent_log_dir:
        # We're a torchrun subprocess (rank ≥ 0 spawned by parent).
        # Reuse parent-resolved log_dir; do NOT re-create or bump run_id.
        log_dir = parent_log_dir
        experiment_id = os.path.basename(log_dir)
    else:
        # Parent / single-process: resolve next run_id and create dir.
        existing = [
            d for d in os.listdir(logs_root)
            if d.startswith(args.experiment_name + "_")
            # Only numeric run suffixes count; sibling dirs like
            # "<name>_smoke" would crash int() below otherwise.
            and d.rsplit("_", 1)[-1].isdigit()
        ]
        next_run_id = (
            max(int(d.rsplit("_", 1)[-1]) for d in existing) + 1
            if existing else 1
        )
        experiment_id = f"{args.experiment_name}_{next_run_id}"
        log_dir = os.path.join(logs_root, experiment_id)
        os.makedirs(log_dir, exist_ok=True)
        # Expose to torchrun-spawned children.
        os.environ["OMNIPIANO_LOG_DIR"] = log_dir

    print(f"Env: {args.env}")
    print(
        f"Mode: {'SMOKE_TEST' if args.smoke_test else 'canonical'} | "
        f"algorithm={args.algorithm} seed={args.seed} "
        f"num_eval_eps={args.num_eval_eps} total_steps={args.total_steps:,} "
        f"vector_env_nums={args.vector_env_nums} device={args.device}"
    )

    # custom_cfgs schema mirrors omnisafe/configs/on-policy/PPOLag.yaml.
    # The yaml default is ``seed: 0``; we override with the protocol's
    # seed so reproducibility is explicit, not inherited. ``gamma`` and
    # ``cost_gamma`` are also overridden because OmniSafe's yaml ships
    # 0.99 while OmniPiano uses 0.8 as a task-property unified across
    # all algorithm backends (see BenchmarkProtocolConfig.gamma).
    cost_gamma = args.cost_gamma if args.cost_gamma is not None else args.gamma

    # Derive OmniSafe save_model_freq (units: epochs) from the protocol's
    # canonical eval cadence (units: env-steps). Each ckpt is then a usable
    # learning-curve sample point for post-hoc deterministic eval.
    #
    # ONE EPOCH IS ALWAYS ``steps_per_epoch`` ENV-STEPS — neither ``parallel``
    # nor ``vector_env_nums`` changes it. Both knobs only decide how those
    # steps are *distributed*: OmniSafe divides the configured value by BOTH
    # to get each rank's per-env rollout length
    # (``policy_gradient.py``: ``steps_per_epoch // world_size //
    # vector_env_nums``), and the step counter is logged from the raw config
    # value (``'TotalEnvSteps': (epoch + 1) * algo_cfgs.steps_per_epoch``).
    # Verified against real runs: a SAC run with ``vector_env_nums=4`` and a
    # PPOLag run with ``parallel=2`` both advance TotalEnvSteps by exactly
    # their configured steps_per_epoch per epoch.
    # (A previous version of this line multiplied by both knobs, which would
    # have doubled the derived interval on the ``parallel=2`` run had it not
    # passed ``--save-model-freq`` explicitly.)
    #
    # CONSEQUENCE — the protocol's 50,000 is NOT reachable on this backend.
    # save_model_freq is quantised to whole epochs, so the achievable
    # checkpoint grid is a multiple of steps_per_epoch: 20k / 40k / 60k … for
    # on-policy. 50,000 is not a multiple of 20,000, and the ``ceil`` below
    # therefore lands on 3 epochs = 60,000. Prefer ``--save-model-freq 1``
    # (20,000 grid): a denser grid can always be subsampled to a coarser one,
    # never the reverse. Truly matching 50,000 would require changing
    # steps_per_epoch itself, i.e. changing PPO's rollout batch size — an
    # algorithm-level change that would also break comparability with every
    # existing OmniSafe run.
    #
    # PPOLag-family steps_per_epoch is 20_000; SAC-family is 2_000 (both are
    # upstream defaults, verified against origin/main of
    # PKU-Alignment/omnisafe). Hard-coded here because OmniSafe doesn't expose
    # it as a Config attribute readable before Agent construction. If a future
    # algo uses a different steps_per_epoch, add it to the dispatch below.
    #
    # User may also pass ``--save-model-freq N`` to override directly.
    _STEPS_PER_EPOCH = 2_000 if args.algorithm in _OFF_POLICY_ALGOS else 20_000
    env_steps_per_epoch = _STEPS_PER_EPOCH
    if args.save_model_freq is not None:
        save_model_freq = int(args.save_model_freq)
    else:
        save_model_freq = max(
            1,
            math.ceil(proto.eval_freq_env_steps / env_steps_per_epoch),
        )

    train_cfgs: Dict[str, Any] = {
        "total_steps": int(args.total_steps),
        "vector_env_nums": int(args.vector_env_nums),
        "parallel": int(args.parallel),
        "device": args.device,
    }
    if args.torch_threads is not None:
        train_cfgs["torch_threads"] = int(args.torch_threads)

    # ``cost_gamma`` lives in algo_cfgs ONLY for on-policy algos (it discounts
    # the cost critic's GAE return). Off-policy algos (SAC / DDPG / TD3 and
    # their Lag / PID variants) don't have a cost_gamma key in their yaml
    # schema and injecting it raises ``KeyError: Invalid key`` at Agent init.
    # ``gamma`` is in algo_cfgs for both.
    algo_cfgs_block: Dict[str, Any] = {"gamma": float(args.gamma)}
    if args.algorithm not in _OFF_POLICY_ALGOS:
        algo_cfgs_block["cost_gamma"] = float(cost_gamma)

    custom_cfgs = {
        "seed": int(args.seed),
        "train_cfgs": train_cfgs,
        "algo_cfgs": algo_cfgs_block,
        "logger_cfgs": {
            "log_dir": log_dir,
            "use_wandb": False,
            "use_tensorboard": True,
            # save_model_freq lives in logger_cfgs (not algo_cfgs) per
            # ``omnisafe/configs/on-policy/PPOLag.yaml:88``. Misplacement
            # raises ``KeyError: Invalid key`` at Agent init.
            "save_model_freq": save_model_freq,
        },
    }

    # Route the constraint hparams to the section their schema lives in. The
    # split was determined by grepping ``omnisafe/configs/**/*.yaml`` —
    # see module-level _LAGRANGIAN_ALGOS / _TRUST_REGION_CONSTRAINED_ALGOS.
    #
    # cost_limit is ALWAYS injected (it defines the safety budget — a task
    # property, not an algo internal hparam). lagrangian_multiplier_init and
    # lambda_lr inherit OmniSafe yaml defaults unless user explicitly passes
    # them — this keeps eval_summary.json/hparams clean (only records what the
    # user actually changed, not echoed library defaults).
    # Off-policy ``update_iters`` + ``buffer_size`` route — must come BEFORE
    # the constraint dispatch below because SACLag is BOTH off-policy AND
    # Lagrangian.
    if args.update_iters is not None:
        if args.algorithm not in _OFF_POLICY_ALGOS:
            print(
                f"[run_omnisafe_template] WARNING: --update-iters is off-policy only; "
                f"algorithm {args.algorithm!r} is on-policy → ignoring."
            )
        else:
            custom_cfgs["algo_cfgs"]["update_iters"] = int(args.update_iters)
    if args.buffer_size is not None:
        if args.algorithm not in _OFF_POLICY_ALGOS:
            print(
                f"[run_omnisafe_template] WARNING: --buffer-size is off-policy only; "
                f"algorithm {args.algorithm!r} is on-policy → ignoring."
            )
        else:
            # ``size`` in off-policy algo_cfgs IS the replay buffer capacity
            # (see ``omnisafe/common/buffer/vector_offpolicy_buffer.py:67``).
            custom_cfgs["algo_cfgs"]["size"] = int(args.buffer_size)

    if args.algorithm in _LAGRANGIAN_ALGOS:
        lag_cfg = {"cost_limit": float(args.cost_limit)}
        if args.lagrangian_multiplier_init is not None:
            lag_cfg["lagrangian_multiplier_init"] = float(args.lagrangian_multiplier_init)
        if args.lambda_lr is not None:
            lag_cfg["lambda_lr"] = float(args.lambda_lr)
        custom_cfgs["lagrange_cfgs"] = lag_cfg
    elif args.algorithm in _TRUST_REGION_CONSTRAINED_ALGOS:
        custom_cfgs["algo_cfgs"]["cost_limit"] = float(args.cost_limit)
        # Note: --lagrangian-multiplier-init / --lambda-lr are silently
        # ignored here — they have no analogue in trust-region constrained
        # algos.
    else:
        # Vanilla unconstrained algo (PPO / SAC / TRPO / DDPG / TD3 / Saute /
        # Simmer / ...). cost_limit etc. don't apply. Warn so a benchmark run
        # with --algorithm PPO on a safety task doesn't silently lose the
        # constraint.
        print(
            f"[run_omnisafe_template] WARNING: algorithm {args.algorithm!r} is "
            "neither Lagrangian-family nor trust-region-constrained — "
            "--cost-limit / --lagrangian-multiplier-init / --lambda-lr are "
            "ignored. Running on a safety task will degenerate to vanilla "
            "unconstrained RL with cost-signal-blind policy."
        )

    agent = omnisafe.Agent(args.algorithm, args.env, custom_cfgs=custom_cfgs)
    print(f"OmniSafe {args.algorithm} training -> {log_dir}")
    agent.learn()

    # OmniSafe writes a TWO-level subdir tree under log_dir:
    #   <log_dir>/<Algo>-{<env_id>}/seed-<NNN>-<timestamp>/torch_save/epoch-*.pt
    # The Evaluator needs the second level (the seed-... dir). Walk both levels.
    def _latest_subdir(d):
        kids = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
        return max(kids, key=os.path.getmtime) if kids else d
    algo_dir = _latest_subdir(log_dir)
    save_dir = _latest_subdir(algo_dir)

    print(f"Running final benchmark eval -> {save_dir}")
    result = _final_eval(save_dir, args.num_eval_eps, train_seed=int(args.seed))

    # Audit metadata — parallel to SB3 templates' eval_summary.json so
    # cross-framework reports diff cleanly.
    result["env"] = args.env
    result["seed"] = int(args.seed)
    result["total_env_steps"] = int(args.total_steps)
    result["num_eval_eps"] = int(args.num_eval_eps)
    result["protocol_version"] = proto.protocol_version
    result["metrics_protocol_version"] = proto.metrics_protocol_version
    result["eval_seed_offset"] = int(proto.eval_seed_offset)
    result["algorithm"] = f"{args.algorithm} (OmniSafe)"
    result["smoke_test"] = args.smoke_test
    hparams = {
        "vector_env_nums": int(args.vector_env_nums),
        "device": args.device,
        "gamma": float(args.gamma),
    }
    # Only record cost_gamma for algos that actually consumed it (on-policy);
    # off-policy yaml has no cost_gamma key, so reporting it would mislead.
    if args.algorithm not in _OFF_POLICY_ALGOS:
        hparams["cost_gamma"] = float(cost_gamma)
    # Record off-policy UTD-defining knobs when applicable so the audit JSON
    # tells reviewers whether this run matches the SB3 UTD=1 convention.
    if args.algorithm in _OFF_POLICY_ALGOS and args.update_iters is not None:
        hparams["update_iters"] = int(args.update_iters)
    # hparams records only what was effectively overridden — cost_limit always
    # (task budget), lagrangian_multiplier_init / lambda_lr only when user
    # passed them. Reviewer reading the audit JSON sees "did this run change
    # anything from paper PPOLag defaults?" at a glance.
    if args.algorithm in _LAGRANGIAN_ALGOS:
        hparams["cost_limit"] = float(args.cost_limit)
        if args.lagrangian_multiplier_init is not None:
            hparams["lagrangian_multiplier_init"] = float(args.lagrangian_multiplier_init)
        if args.lambda_lr is not None:
            hparams["lambda_lr"] = float(args.lambda_lr)
    elif args.algorithm in _TRUST_REGION_CONSTRAINED_ALGOS:
        hparams["cost_limit"] = float(args.cost_limit)
    result["hparams"] = hparams

    out_path = os.path.join(log_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"eval summary -> {out_path}")


if __name__ == "__main__":
    main()
