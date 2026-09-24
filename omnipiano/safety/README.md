# OmniPiano safety experiments

This directory is integrated with `main` based on commit `ce298d8`. No changes to
the main configuration, registry, wrappers, tasks, examples, or dependencies are required.
The existing `constraints.py` is preserved for compatibility with main's legacy registrations.
The isolated installation profile and its NumPy 1.x compatibility guard are
documented in [Safety Quick Start](QUICKSTART.md); no root files are patched.
See [Main integration](MAIN_INTEGRATION.md) for the merge scope and validation.

To run additional native OmniSafe methods, see [Algorithm interface](ALGORITHMS.md):
32 online model-free choices (the original five plus 27), with explicit task,
algorithm and seed selection. The original paper matrices remain unchanged.

## Safety semantics × cost setting

Let `e_i >= 0` be normalized excess for each measurement unit, and `N` the fixed
number of selected units. The default weight is 1:

- **Event:** `1[any e_i > 0]` per control step, not a count of threshold crossings.
- **Excess:** `mean(e_i)` per control step.
- **Fraction:** `mean(1[e_i > 0])` per control step. The illustration's “Friction”
  means **Fraction**, not physical friction or a temporal violation rate.

| Semantic | Unit / signal | Excess and default parameters | Settings |
|---|---|---|---|
| Joint range | THJ2/FFJ2/MFJ2/RFJ2/LFJ2 on every hand; physical qpos | Distance outside central 50% of native range, divided by full native span; values ≤ 1e-8 treated as zero | Event, Excess, Fraction |
| Actuator power | Event/Excess: total power over selected hands; Fraction: total power of each selected hand separately | Event/Excess: `max(P_total / P_ref - 1, 0)`, default `P_ref = 4 × number_of_hands`; Fraction: share of selected hands above the per-hand reference, default 4 | Event, Excess, Fraction |
| Injured finger | Five named thumb actuators THJ1–THJ5 on one fixed right hand | Per-actuator `max(P_i / 1.0 - 1, 0)` | Event, Excess, Fraction |
| Hand collision | Every unordered pair of selected hands, including zero-contact pairs | Sum of absolute normal contact forces within each pair, then `max(F_pair / 10 - 1, 0)` | Event, Excess, Fraction |

This gives **4 semantics × 3 settings = 12 combinations**. Power Fraction is
`number_of_selected_hands_with_P_hand_above_reference / number_of_selected_hands`.
Equality to the reference is safe. Every actuator on a selected hand contributes
to that hand's power; the denominator counts hands, not actuators or time steps.
The weight is applied after taking the fraction.

The three power settings share the power semantic but **not the same measurement
unit**: Event/Excess retain total-system power; Fraction measures local overload
coverage. This compatibility choice keeps the original main experiment unchanged.
It is not a strict aggregation-only ablation. With hand powers `[6, 0]`, a total
reference of 8 yields Event=0 and Excess=0, while a per-hand reference of 4 yields
Fraction=0.5. When constructing a power/Fraction CostSpec, `reference` means the
per-hand limit; it is not automatically divided by the number of hands.

For two hands, collision also has
only one pair, so Event and Fraction are identical; both are retained to keep
the same pair-based definition for 3–5 hands. A pair can have multiple contact
points; they are summed before thresholding. Self-contact and piano contacts
are excluded; tangential force is not included.

The protected hand is `rh` at 2/3 hands and `rh_t` at 4/5 hands. These are soft
costs, not actual anatomical injuries, disabled actuators, or physical joint clamps.
Static force at zero actuator velocity has zero power. Power is not electrical
energy, and episode sums have no `dt` factor. No physical initialization masking
is applied. Thresholds and budgets are candidate engineering settings, not safety
standards or guarantees of feasibility/learnability.

The new injured-finger cost has a per-actuator threshold and differs from the old
unthresholded `injured_finger_load`. Collision Excess averages all hand pairs,
not the old maximum-pair continuous `inter_hand_contact`. Their results must not
be relabelled as equivalent. `action_slew` and the retired `power_spike` task are
not migrated. Power/Event is explicitly a new setting in this factorized catalogue;
it does not reinstate the old task IDs or validate their thresholds.

## Registration

```python
import omnipiano
from omnipiano.safety.suite import MAIN, register_task

env_id = register_task(MAIN[0])
env = omnipiano.make(env_id, seed=1)
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info['step_safety/cost_total'])
env.close()
```

Registration is explicit and process-local. No import side effect changes main's
legacy environments. `register_all()` returns all unique new tasks. Cost is
computed after the physical step and flows through the unmodified SafetyWrapper.
`Task.budget` is the episode budget; main's SafetyConfig has no budget field.
`safety.runtime.algorithm_config` routes this same value to the appropriate algorithm.

All tasks use OT fingering reward. A safety-local `SafetyEnvConfig` subclass exposes
the upstream `energy_penalty_coef=0.0` parameter through main's existing dataclass
forwarding, maintaining the previous safety protocol without editing main.
Other default reward components (including forearm shaping) remain unchanged.
Cost is never subtracted from reward or used for early termination. PPO ignores
the extra cost in optimization but is not necessarily blind to related reward terms.

## Main experiment: previous eight selections retained

| Index of environment | Hands | Song | Semantic / setting | Episode budget |
|---|---:|---|---|---:|
| 0 | 2 | ForElise | Joint range / Fraction | 19.95 |
| 1 | 2 | ClairDeLune | Joint range / Excess | 11.76 |
| 2 | 3 | PicturesGreatKiev | Actuator power / Excess | 14.4 |
| 3 | 3 | PolonaiseOp40No1 | Actuator power / Excess | 11.26 |
| 4 | 4 | PicturesGreatKiev | Joint range / Excess | 14.4 |
| 5 | 4 | PicturesGreatKiev | Actuator power / Excess | 14.4 |
| 6 | 5 | PicturesGreatKiev | Joint range / Event | 36 |
| 7 | 5 | PicturesGreatKiev | Joint range / Excess | 14.4 |

The main experiment deliberately covers the previous range/power selections,
not all four new semantics. New IDs distinguish this main-compatible code version
from previous local versions; preserving selections is not bitwise reproduction.

| Group | Design | Runs | Steps per run |
|---|---|---:|---:|
| `main` | 8 environments × 5 algorithms × seeds 1,2,3 | 120 | 5M |
| `hands` | 4 nested hand counts × 5 algorithms × seeds 1,2,3 | 60 | 5M |
| `budget` | 5 budgets × 4 constrained algorithms × 3 seeds + 3 PPO references | 63 | 5M |
| `extensions` | 4 hand/song anchors × 12 settings × PPO/PPOLag × seed 0 | 96 | 2M |

The first three groups are **243 independent runs**. No cross-device/group model,
result or state reuse. The optional 96-run extension screen includes old-axis
controls as well as new semantic candidates; it is not part of the main results.
Its anchors are 2H ForElise, 3H PolonaiseOp40No1, and 4/5H PicturesGreatKiev.
There are now 48 extension environments and 58 unique registered environments
after combining main and ablations. The new four power/Fraction tasks use per-hand
reference 4 and candidate episode budgets 19.95, 28.15, 36, 36 respectively.
These budgets follow the existing `0.05 × nominal_episode_steps` Fraction rule;
they have not been calibrated to make a particular algorithm succeed.

Manifest version is now `safety-semantic-20260919-v2`. Original main/hand/budget
environment IDs, formulas, budgets and run selections are unchanged, but the code
fingerprint and extension cell indices changed. Do not switch versions mid-batch;
pin commit `303e68b` when completing an existing v1 batch, or use a new output
directory for v2. Do not relabel historical v1 validation as testing this addition.

Hand sensitivity fixes Great Kiev, total-power Excess, `P_ref=16`, and budget 14.4.
Its nested order is `rh_b, lh_t, lh_b, rh_t, rh_c`; each added hand preserves existing
hands' poses. The four-hand order differs from the original main task, so their
checkpoints must not be reused. This is a morphology sensitivity test, not proof
that extra hands alone cause every observed change in optimization difficulty.

Budget sensitivity fixes the original/default four-hand Great Kiev task and
`P_ref=16`, with episode budgets `1.44, 4.32, 14.4, 43.2, 144`.
It changes the CMDP budget, **not** the physical power threshold. Each budget has
a distinct environment ID. A PPO reference is trained locally at the default budget.

## Installation and execution

Follow [Safety Quick Start](QUICKSTART.md) to create a **separate Python 3.10
Conda environment**, install the pinned safety dependencies and preprocess PIG.
Do not install OmniSafe directly into the general benchmark's environment.
The guide documents the upstream Safety-Gymnasium/MuJoCo dependency exception.
After installation, run from the repository root:

```bash
python -m pytest omnipiano/safety/tests -q
python -m omnipiano.safety.smoke --steps 2 --out safety_smoke.json
```

Training defaults: 5M steps, reward/cost gamma 0.8, one environment, one Torch
thread, observation normalization on, reward/cost normalization off. The five
algorithms are native OmniSafe **PPO, PPOLag, OnCRPO, CPO, CUP**, with official
0.5.0 algorithm-specific defaults retained. PPOLag/CUP use `lagrange_cfgs.cost_limit`;
OnCRPO/CPO use `algo_cfgs.cost_limit` and explicitly enable their cost critic.
Networks can use `cuda:0`; MuJoCo simulation remains CPU based.

```bash
# Dry-run plans only; no training unless --execute is present.
python -m omnipiano.safety.run --group main --out results_main --device cuda:0
python -m omnipiano.safety.run --group hands --out results_sensitivity --device cuda:0
python -m omnipiano.safety.run --group budget --out results_sensitivity --device cuda:0

# Minimal algorithm test: one selected cell, 2K steps and one eval episode/checkpoint.
python -m omnipiano.safety.run --group main --index 0 --out smoke_runs --smoke-test --device cuda:0 --execute

# Formal batches, on the respective devices. Groups remain independent.
python -m omnipiano.safety.run --group main --out results_main --device cuda:0 --execute
python -m omnipiano.safety.run --group hands --out results_sensitivity --device cuda:0 --execute
python -m omnipiano.safety.run --group budget --out results_sensitivity --device cuda:0 --execute
```

`--index` selects a **zero-based run cell**, not the environment-row index above.
Main order is environment, algorithm (`PPO, PPOLag, OnCRPO, CPO, CUP`), then seed.
For the first environment, smoke-test algorithm indices are 0,3,6,9,12.
Omitting `--index` runs the entire selected group sequentially when `--execute` is set.

Each run records task config, resolved budget, code/MIDI hashes, dependency versions,
seed, device, checkpoint/evaluation schedule and algorithm overrides. Failed training
does not automatically resume optimizer state or overwrite partial logs. Replay failure
retains status `trained`; rerunning retries only evaluation. Exclusive per-run locks
prevent two processes writing the same cell. Inspect any stale lock before manually
removing it. Completed runs are skipped. Keep source/dependencies fixed during a batch.

## Evaluation and figures

Training rollout curves are OmniSafe `progress.csv` (reward and episode cost).
Formal models are saved at step 0 and every 20K steps: 251 checkpoints at 5M.
After training, every checkpoint receives 10 deterministic evaluation episodes,
with seeds `train_seed + 10000 + episode_index`. This is checkpoint replay,
not evaluation interleaved with training. Evaluation steps do not count toward 5M.
F1 is the existing environment's episode metric; no new F1 definition is introduced.
Repeated deterministic evaluation episodes are not independent training seeds.

```bash
python -m omnipiano.safety.plot --group main --runs results_main --out figures_main
python -m omnipiano.safety.plot --group hands --runs results_sensitivity --out figures_sensitivity
python -m omnipiano.safety.plot --group budget --runs results_sensitivity --out figures_sensitivity
# Separate training-rollout reward/cost figures:
python -m omnipiano.safety.plot --group main --runs results_main --out figures_main --source rollout
```

Replay output: reward, F1, cost in separate PNG/PDF figures. Main: 8 panels with all
5 algorithms. Hands: 4 panels (one constrained algorithm each), four hand-count
curves and four matching PPO dashed references. Budget: 4 panels, five budget
curves and one PPO reference each. Bands are mean ± sample SD across exactly
training seeds 1,2,3, with no interpolation or mixing incomplete/2M/smoke results.
F1 is available for replay, not fabricated for the training-rollout logs.

Environment quality should consider cost learning, reward/F1 trade-offs, constraint
activation, morphology coverage and algorithm separation—not only final feasibility.
Passing the smoke tests establishes executability, not a successful benchmark result.
