# WinterWind SCHO 5 x 5 x 3 reproduction suite

This directory freezes the paper comparison at reviewed runtime commit
`f33ec7cff1591e374fc1be5de78a643b832291b6`.

The matrix contains IPPO, MAPPO, HAPPO, A2PO and FACMAC on Base,
Observability, Coupling, Heterogeneity and Scalability. Seeds 0, 1 and 2 yield
75 training runs. Every run uses 10M environment steps, `gamma=0.8`, collision
penalty coefficient 0.1, evaluation every 50k steps and evaluation video every
500k steps plus the final policy.

Contents:

- `runs/`: the 75 complete, portable training JSON files;
- `tasks/`: the five task definitions without algorithm duplication;
- `manifest.json`: identities and canonical JSON hashes;
- `reproduction_plan.json`: the 75 expected algorithm/task/seed cells;
- `provenance.json`: reviewed code, dataset and environment provenance;
- `requirements-reproduction.txt`: the reference Python environment;
- `dataset_identity.json`: licensed PIG v1.2/WinterWind identity;
- `checksums.sha256`: raw-file integrity checks;
- `reproduce.sh`: validation, runtime environment audit and queue launch.

From the repository root, a conservative six-GPU launch is:

```bash
PYTHON=/root/miniconda3/envs/pianist/bin/python \
RUN_ROOT=/root/autodl-fs/omnipiano_runs/scho_winterwind_5x5x3 \
bash omnipiano/multiagent/configs/scho_winterwind_5x5x3/reproduce.sh \
  --gpus 0 1 2 3 4 5 \
  --slots-per-gpu 1 \
  --object-store-mb 4096 \
  --stagger 60
```

Use `--verify-only` as the first argument to perform all static checks and the
five-task MuJoCo audit without launching training. Set
`STRICT_DEPENDENCIES=1` to require exact equality with the recorded package
versions. The frozen JSON files contain `run_dir: null`; `reproduce.sh` injects
`RUN_ROOT` only into per-run state snapshots, leaving source checksums intact.

The PIG dataset cannot be redistributed. The script requires the official
preprocessed `EtudeOp25No11` resource before launching and reports its SHA-256
in the generated environment audit logs. `dataset_identity.json` therefore
records the canonical identity but does not invent a hash on a machine where
the licensed MIDI is absent.
