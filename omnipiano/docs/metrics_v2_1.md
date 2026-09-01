# Metrics Protocol v2.1: definition, migration, and audit guide

Metrics-v2.1 is a proposed corrective protocol implemented on this branch for
OmniPiano's reward-independent scorecard. It does **not** change the
environment reward, trained policies,
musical event matcher, or the original trace bytes. It fixes contact-force
attribution, makes invalid physical channels fail visibly, and records every
scoring parameter needed to reproduce a result.

Runtime evaluation on this branch uses Benchmark Protocol 1.1: MIDI silence
trimming takes global note-time extrema, and final/replay evaluation episode
`i` uses `training_seed + 10_000 + i` in every supported backend. Periodic SB3
evaluation is seeded at `training_seed + 10_000`; the formal protocol default
is one episode per checkpoint. This training/runtime protocol is versioned
independently from Metrics-v2.1 scorer semantics.

## What is a paper metric

The primary musical outcomes remain:

- note-event precision, recall, and F1 at a 50 ms onset tolerance;
- key-time micro precision, recall, and F1;
- onset, offset, and duration error;
- false positives during target-rest frames;
- sustain precision, recall, and F1.

Reward is reported as an algorithm/environment return, not as the benchmark's
sole measure of musical performance. The historical frame-macro `f1` remains
for backward compatibility but is not a headline metric because silent frames
can dominate it.

The cooperation/physical scorecard additionally reports attribution by hand
and physical player, active/effective players, contribution balance,
collisions, actuator work, and their validity gates.

## Versioned scoring constants

| Parameter | Metrics-v2.1 value | Unit | Rule |
|---|---:|---|---|
| Event onset tolerance | 0.05 | s | pitch-exact, one-to-one event match |
| Contact attribution window | 0.05 | s | half-open `[onset, onset + 0.05)` |
| Contact participation floor | `1e-6` | N | peak force must be strictly greater |
| Collision force floor | `1e-6` | N | force must be strictly greater |
| Motor-active power floor | `1e-6` | W | power must be strictly greater |

The `1e-6 N` contact value is a **numerical floor**, not a calibrated piano
actuation threshold. It removes a separated cluster of simulator denormals
around `1e-323` to `1e-311 N`. A later physical calibration must receive a new
metrics protocol version rather than silently changing this constant.

For each actual note event:

1. A hand participates if its peak fingertip-to-key normal force in the
   half-open onset window is greater than `1e-6 N`.
2. The primary hand is the participating hand with maximum integrated impulse.
   A nonparticipant can never become primary merely by accumulating sub-floor
   samples.
3. The primary physical player is obtained from the versioned hand-to-player
   assignment. Monolithic N-hand PPO uses the same virtual physical-player
   groups as its IPPO/MAPPO counterpart; this is a morphology comparison, not
   a claim that the monolithic policy contains multiple policy modules.

## Invalid-channel semantics

Normal contact/collision force is physically non-negative. Any NaN, infinity,
or finite negative value in the relevant scope fails its quality gate.

- A bad contact sample invalidates that note event and the trace-level
  attribution claim. Attribution-derived hand/player scalars become missing;
  independent musical metrics remain valid.
- A bad collision sample invalidates rate, event count, force integral, and
  maximum force together. The old mixed state—finite-looking counts alongside
  a null force integral—is forbidden.
- A nonfinite or finite-negative power sample invalidates all work and
  motor-activity metrics. Musical and attribution metrics remain available.
- Missing physical arrays are explicitly marked unavailable and round-trip as
  `None`; they are never serialized as an unreadable object array.

The audit CSV gives invalid contact events their own
`invalid_contact_event` stratum. They are not mislabeled as ordinary
unattributed notes. Force summaries for an invalid event are left blank.

Territory recall is also corrected: `correct_events` records every matched
event attributed to an entity, while `eligible_correct_events` is the subset
inside that entity's registered territory. `target_recall` uses
`eligible_correct_events / eligible_target_events`, so it cannot exceed one.

## Evidence from the frozen 36-run MARL matrix

Source traces:

`/data/giil/zhangzy/omnipiano/runs/marl_core_v1/formal`

The matrix contains 36 final evaluations: 3 algorithms × 4 pieces × 3 seeds.
It has 2,086 actual note events and 1,398 matched events. All traces use a
50 ms control timestep.

### Threshold sensitivity at the frozen onset-only window

| Force floor | Attributed events | Raw coverage |
|---:|---:|---:|
| `0 N` | 1,794 | 86.00% |
| `1e-12 N` | 1,197 | 57.38% |
| `1e-9 N` | 1,191 | 57.09% |
| `1e-7 N` | 1,186 | 56.86% |
| `1e-6 N` | 1,186 | 56.86% |
| `1e-5 N` | 1,185 | 56.81% |
| `1e-3 N` | 1,179 | 56.52% |
| `1e-2 N` | 1,160 | 55.61% |

At `1e-6 N`, 1,186 of 2,082 sensor-valid events are attributed (56.96%
conditional coverage). Four events contain a nonfinite sample in their
relevant onset/key window and are excluded by the validity gate.

The zero-floor result gains 608 events solely because any positive value,
including denormals, counts as contact. Among the 1,186 events attributed by
both configurations, the primary hand and primary player agree in every case.

### Window sensitivity at `1e-6 N`

| Inclusive diagnostic frame offsets | Attributed events | Raw coverage | Primary-player flips among common events |
|---|---:|---:|---:|
| `[0, 0]` | 1,186 | 56.86% | 0 |
| `[-1, 0]` | 1,499 | 71.86% | 0 |
| `[0, +1]` | 1,660 | 79.58% | 0 |
| `[-1, +1]` | 1,801 | 86.34% | 0 |
| `[-2, +2]` | 1,947 | 93.34% | 0 |

Identity is stable for events shared with the reference window, but coverage
is strongly window-dependent. Therefore v2.1 keeps the frozen 50 ms
onset-only window to isolate the attribution-threshold correction from a
separate window-definition change; the paper must not claim that it is
physically optimal until replay videos have been manually reviewed.

### Quality-gate failures found by rescoring

Contact attribution is invalid in 4/36 final traces:

- IPPO / GreatKiev / seed 1;
- IPPO / Sonata30 / seed 2;
- MAPPO / GreatKiev / seed 0;
- monolithic PPO / Sonata30 / seed 1.

Collision metrics are invalid in 5/36 final traces:

- IPPO / GreatKiev / seed 1;
- MAPPO / Sonata30 / seed 0;
- MAPPO / WinterWind / seed 1;
- monolithic PPO / GreatKiev / seed 0;
- monolithic PPO / WinterWind / seed 2.

Power metrics pass in 36/36. All reward-independent musical invariants pass
the migration check. These failures are measurement-quality findings; they do
not imply that the corresponding policy's note-event F1 is zero.

## Immutable offline migration

Never overwrite a Metrics-v2.0 run or relabel its W&B record. Recompute into a
new directory:

```bash
python -m omnipiano.benchmark.rescore_marl \
  --input-root /data/giil/zhangzy/omnipiano/runs/marl_core_v1/formal \
  --output-root /data/giil/zhangzy/omnipiano/runs/marl_core_v1/postprocess/metrics_v2_1_final
```

The rescorer is transactional. A failure in any run removes the temporary
tree, leaving no partial destination. Every manifest records:

- hashes of the four source artifacts: trace, trace metadata, episode
  scorecard, and summary;
- Git commit/dirty state, Python and NumPy versions, and hashes of the six
  in-repo scoring/export implementation files used by the rescorer;
- every threshold/tolerance above;
- hashes of the five generated card, summary, audit, metadata, and checklist
  payloads (the manifest and tree summary are not self-hashed);
- the per-metric diff from Metrics-v2.0 and all quality gates.

It rechecks the source hashes after writing and preserves the legacy musical
compatibility fields. The 36-run tree summary aggregates added/changed metric
counts and quality failures.

Reproduce threshold and window sensitivity separately:

```bash
python -m omnipiano.benchmark.attribution_sensitivity \
  --input-root /data/giil/zhangzy/omnipiano/runs/marl_core_v1/formal \
  --output /data/giil/zhangzy/omnipiano/runs/marl_core_v1/postprocess/metrics_v2_1_sensitivity_final.json
```

## Online evaluation, CSV, and W&B

The SA and PettingZoo MA terminal `info` dictionaries on this branch receive
the full namespaced scorecard. Final evaluation cards and trace metadata are
tagged `metrics_protocol_version=2.1`; SA/SB3 online-evaluation and OmniSafe
checkpoint-replay CSVs use one shared schema and carry the same version.

Those SA/OmniSafe episode CSVs include headline
music/cooperation/physical columns, every quality gate and threshold,
morphology-specific hand/player JSON, and a strict JSON copy of the complete
benchmark scorecard. MA evaluation emits the equivalent versioned JSONL/JSON
cards and audit artifacts rather than an episode CSV. JSON nonfinite values
become `null`; they must remain missing when uploaded to W&B and must never be
filled with zero. Existing v2.0 W&B runs remain historical records; v2.1
rescoring should use a distinct job type/group or project namespace.

Contact-force, collision, and actuator-power capture scans MuJoCo state every
step. SA environments therefore disable those three channels in `mode="train"`
and enable them in `mode="eval"`. The MA factory defaults to training behavior;
`evaluate_policy` explicitly enables `metrics_capture_physics`. Disabled
channels are marked unavailable (`None` plus an availability gate), never
filled with zero. Formal scorecards must come from evaluation mode.

## Required manual review before a paper claim

Each rescored run deterministically selects up to 12 events across matched,
unmatched, attributed, unattributed, and invalid-contact strata. Across 36
runs this yields at most 432 review items. Reviewers should seek to the stated
MP4 frame/time and record whether the visible touching hand agrees with the
reported primary hand.

Video is a sanity check for hand identity and temporal alignment, not ground
truth for force magnitude. A force threshold still requires simulator-channel
validation or an external physical calibration study.

## Acceptance gates

Before merging or launching new formal runs:

1. Pure metric, exporter, rescorer, checklist, transaction, and CSV tests pass.
2. A real SA episode emits all Metrics-v2.1 terminal keys and a schema-aligned
   CSV.
3. A real MA episode exports trace, actions, checklist, and versioned card.
4. Musical invariants pass on all 36 frozen final traces.
5. Input and output hashes verify.
6. One replay video passes manual contact-attribution review before scaling the
   replay job to all 36 runs.

Running Robustness jobs that started from the frozen Metrics-v2.0 snapshot are
allowed to finish unchanged. They must be evaluated/rescored under an explicit
v2.1 label later; changing their live source mid-run would destroy provenance.
