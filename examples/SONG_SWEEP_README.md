# Song sweep — PPO / SAC on the RoboPianist etude pieces (branch `song-sweep-ot`)

## Goal

We want to know **for which pieces the shaped training reward is a good proxy
for F1**. For every piece we train one PPO run and one SAC run (seed 0, 5M
env steps each, SB3 library defaults), then look at the reward-vs-F1
relationship of the periodic evaluations. You only need to produce the runs;
the analysis is done on our side from the log directories you send back.

Task setting: **2 hands, OT (optimal-transport) fingering reward, no safety
constraints**. All env ids follow `OmniPiano-{Piece}-FingeringOT-v0`.

## What to run — 9 pieces × 2 algos = 18 runs

| # | Piece | env id | episode steps |
|---|-------|--------|---------------|
| 1 | Twinkle Twinkle (Rousseau) | `OmniPiano-TwinkleTwinkleRousseau-FingeringOT-v0` | 158 |
| 2 | Piano Sonata No. 23, 2nd mov | `OmniPiano-PianoSonataNo232NdMov-FingeringOT-v0` | 313 |
| 3 | Piano Sonata No. 2, 1st mov | `OmniPiano-PianoSonataNo21StMov-FingeringOT-v0` | 361 |
| 4 | Partita No. 2, 6th mov | `OmniPiano-PartitaNo26-FingeringOT-v0` | 425 |
| 5 | French Suite No. 1 Allemande | `OmniPiano-FrenchSuiteNo1Allemande-FingeringOT-v0` | 479 |
| 6 | Golliwogg's Cakewalk | `OmniPiano-GolliwoggsCakewalk-FingeringOT-v0` | 492 |
| 7 | Bagatelle Op. 3 No. 4 | `OmniPiano-BagatelleOp3No4-FingeringOT-v0` | 550 |
| 8 | Piano Sonata K279, 1st mov | `OmniPiano-PianoSonataK279InCMajor1StMov-FingeringOT-v0` | 579 |
| 9 | Kreisleriana Op. 16 No. 8 | `OmniPiano-KreislerianaOp16No8-FingeringOT-v0` | 585 |

For each env id run **both** `--algo ppo` and `--algo sac`, `--seed 0`,
`--total-steps 5000000`. Do not change any other hyper-parameter (see
"Do not change" below).

Four pieces from the RoboPianist website are intentionally *not* in the list
because their episodes exceed 650 steps (Sonata D845 1st, French Suite No. 5
Sarabande / Gavotte, Waltz Op. 64 No. 1). Skip them.

## Setup

```bash
git clone <repo> && cd SafeRoboPianist
git checkout song-sweep-ot
# conda env + system deps: follow README.md "Quick Start" (python 3.10,
# fluidsynth, ffmpeg, then `pip install -e .`)
export MUJOCO_GL=egl          # headless rendering; the scripts set this too
```

Versions used on our side (please report yours in the results):
`stable-baselines3 2.7.1`, `sb3-contrib 2.7.1`, `torch 2.10.0`.

Sanity check before the real runs (≈1–2 min):

```bash
python examples/run_sb3_baseline.py --algo ppo \
  --env OmniPiano-PianoSonataNo232NdMov-FingeringOT-v0 \
  --experiment-name smoke_sweep --seed 0 --smoke-test
python examples/run_sb3_baseline.py --algo sac \
  --env OmniPiano-PianoSonataNo232NdMov-FingeringOT-v0 \
  --experiment-name smoke_sweep --seed 0 --smoke-test
rm -rf examples/logs/smoke_sweep*
```

Both should end by printing an `eval_summary` JSON block with `return_mean`
and `length_mean: 313.0`.

## How to run

### Option A — the launcher script (sequential)

```bash
examples/runs/song_sweep_ot.sh              # all 9 pieces, PPO then SAC
examples/runs/song_sweep_ot.sh ppo          # PPO only
examples/runs/song_sweep_ot.sh sac Partita  # SAC only, pieces whose name contains "Partita"
```

It runs the 18 jobs **one after another** and writes
`examples/logs/{algo}_sweep_{piece}_seed0_1/` plus a console log
`examples/logs/{algo}_sweep_{piece}_seed0.log`. The piece list is read from
`omnipiano.envs._SONG_SWEEP_OT_PIECES`, so it always matches the table above.

### Option B — one job at a time (for your own parallelism)

```bash
python examples/run_sb3_baseline.py \
  --algo ppo \
  --env OmniPiano-PartitaNo26-FingeringOT-v0 \
  --experiment-name ppo_sweep_partitano26_seed0 \
  --seed 0 --total-steps 5000000
```

Please keep the `--experiment-name` pattern
`{algo}_sweep_{piece_lowercase}_seed{seed}` so directories line up with ours.

### Parallelism — your call

Every PPO job spawns **16** MuJoCo subprocesses, every SAC job **24** (these
are the library-default `n_envs` on our 16C/32T, 60 GB box). On our machine
starting several jobs at the same instant sometimes OOMs during env
construction, which is why the launcher is sequential. Your hardware is
different, so decide yourself how many jobs to run concurrently — e.g. start a
second copy of the launcher with a disjoint piece filter, or use Option B
under `nohup`/tmux. If you do run concurrently, stagger the launches by a
minute or two.

Rough cost on our box: one 5M-step PPO run ≈ 4–6 h, one SAC run ≈ 10–14 h.

### Do not change

- `--n-envs` for SAC. It is **not** a pure speed knob: SB3's auto-tuned
  entropy coefficient depends on how many transitions arrive per update.
  We measured α collapsing at n_envs ≤ 8 and exploding at n_envs ≥ 32; 24 is
  what all our other SAC results use. If your machine cannot afford 24
  subprocesses, run fewer jobs concurrently rather than lowering `--n-envs`.
- `--n-envs` for PPO (16). Less sensitive, but it changes the rollout batch.
- `--gamma`, `--eval-freq` (50 000), `--num-eval-eps` (1), `--learning-starts`,
  `--ent-coef`, `--gradient-steps` — leave all at their defaults.
- Anything under `omnipiano/`.

If a run crashes, restart it from scratch with the same command (there is no
resume); the new directory gets suffix `_2`, delete the broken `_1`.

## What to send back

For each of the 18 runs, the whole log directory
`examples/logs/{algo}_sweep_{piece}_seed0_1/`, minus the two video folders
(`videos/`, `videos_best/`) which are big and not needed. What we use:

- `eval_episode_metrics_*.csv` — per-eval `ep_return`, `ep_f1`, reward
  components (this is the file the reward↔F1 correlation is computed from)
- `evaluations.npz`, `eval_summary.json`
- `tensorboard/` (training-rollout curves)
- `fig_*.png` (quick look)
- the console `.log` if you used the launcher

`zip -r sweep_logs.zip examples/logs/*_sweep_* -x '*/videos/*' '*/videos_best/*'`
is fine. Also note the SB3/torch versions and CPU/GPU used.

## Questions

Ping us if the smoke test fails, a run dies repeatedly on one piece, or
throughput is very different from the estimate above.
