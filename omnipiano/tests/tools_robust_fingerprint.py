"""Capture a bit-exact behavioural fingerprint of every registered robust env.

Run BEFORE and AFTER the per-channel-dist refactor; the two JSON files must
be byte-identical (§0.7 equivalence-gate convention).

For each robust env: fixed seed, fixed action sequence, N steps; record the
per-step injected-noise telemetry plus a hash of the observation and the
received reward — i.e. everything the agent experiences.
"""
import hashlib
import json
import sys

import numpy as np

from omnipiano.envs import registration
from omnipiano.utils.info_keys import InfoKeys

N_STEPS = 8
SEED = 0


def robust_env_ids():
    out = []
    for env_id, spec in registration._registry.items():
        rc = spec.robust_config
        if rc is None:
            continue
        if any(rc.is_channel_active(c) for c in ("action", "obs", "reward")):
            out.append(env_id)
    return sorted(out)


def fingerprint(env_id):
    env = registration.make(env_id, seed=SEED)
    try:
        obs, _ = env.reset(seed=SEED)
        dim = env.action_space.shape[0]
        rng = np.random.default_rng(12345)          # action source, not env RNG
        actions = [rng.uniform(-1, 1, size=dim) for _ in range(N_STEPS)]
        rec = {"obs0_sha": hashlib.sha1(np.asarray(obs).tobytes()).hexdigest()[:16],
               "steps": []}
        for a in actions:
            obs, reward, term, trunc, info = env.step(a)
            rec["steps"].append({
                "act_l2": repr(float(info[InfoKeys.ROBUST_NOISE_ACTION_L2])),
                "obs_l2": repr(float(info[InfoKeys.ROBUST_NOISE_OBS_L2])),
                "rew_noise": repr(float(info[InfoKeys.ROBUST_NOISE_REWARD])),
                "reward": repr(float(reward)),
                "obs_sha": hashlib.sha1(np.asarray(obs).tobytes()).hexdigest()[:16],
            })
            if term or trunc:
                break
        return rec
    finally:
        env.close()


def main():
    out_path = sys.argv[1]
    ids = robust_env_ids()
    print(f"fingerprinting {len(ids)} robust envs x {N_STEPS} steps ...")
    data = {}
    for i, env_id in enumerate(ids, 1):
        data[env_id] = fingerprint(env_id)
        print(f"  [{i}/{len(ids)}] {env_id}")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
