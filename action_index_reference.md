# Action Index Reference

This document maps each index in the action vector to its corresponding actuator and joint.
Use this as a reference when creating tasks with `JointMagnitudeConstraint(index=..., ...)`.

## Action Vector Layout

The full action vector is constructed as:

```
action = [right_hand_actuators, left_hand_actuators, sustain_pedal]
```

Evidence: `piano_with_shadow_hands_test.py` line 111:

```python
action = np.concatenate([right_action, left_action, [0]])
```

Each hand has **20 XML-defined actuators + 2 forearm DOFs = 22 actuators**.
Total action dimension: **22 (right) + 22 (left) + 1 (sustain) = 45**.

## Right Hand (Index 0-21)

| Index | Actuator       | Joint/Tendon | Body Part              |
|-------|----------------|--------------|------------------------|
| 0     | rh_A_WRJ2      | rh_WRJ2      | Wrist yaw              |
| 1     | rh_A_WRJ1      | rh_WRJ1      | Wrist pitch            |
| 2     | rh_A_THJ5      | rh_THJ5      | Thumb base             |
| 3     | rh_A_THJ4      | rh_THJ4      | Thumb proximal         |
| 4     | rh_A_THJ3      | rh_THJ3      | Thumb hub              |
| 5     | rh_A_THJ2      | rh_THJ2      | Thumb middle           |
| 6     | rh_A_THJ1      | rh_THJ1      | Thumb distal           |
| 7     | rh_A_FFJ4      | rh_FFJ4      | Index finger knuckle   |
| 8     | rh_A_FFJ3      | rh_FFJ3      | Index finger proximal  |
| 9     | rh_A_FFJ0      | rh_FFJ0 (t)  | Index finger mid+dist  |
| 10    | rh_A_MFJ4      | rh_MFJ4      | Middle finger knuckle  |
| 11    | rh_A_MFJ3      | rh_MFJ3      | Middle finger proximal |
| 12    | rh_A_MFJ0      | rh_MFJ0 (t)  | Middle finger mid+dist |
| 13    | rh_A_RFJ4      | rh_RFJ4      | Ring finger knuckle    |
| 14    | rh_A_RFJ3      | rh_RFJ3      | Ring finger proximal   |
| 15    | rh_A_RFJ0      | rh_RFJ0 (t)  | Ring finger mid+dist   |
| 16    | rh_A_LFJ5      | rh_LFJ5      | Little finger metacarp |
| 17    | rh_A_LFJ4      | rh_LFJ4      | Little finger knuckle  |
| 18    | rh_A_LFJ3      | rh_LFJ3      | Little finger proximal |
| 19    | rh_A_LFJ0      | rh_LFJ0 (t)  | Little finger mid+dist |
| 20    | forearm_tx      | forearm_tx   | Forearm slide along keyboard |
| 21    | forearm_ty      | forearm_ty   | Forearm slide in/out (depth) |

> **(t)** = tendon-driven actuator coupling two joints (e.g., FFJ0 drives FFJ2 + FFJ1 together).

## Left Hand (Index 22-43)

| Index | Actuator       | Joint/Tendon | Body Part              |
|-------|----------------|--------------|------------------------|
| 22    | lh_A_WRJ2      | lh_WRJ2      | Wrist yaw              |
| 23    | lh_A_WRJ1      | lh_WRJ1      | Wrist pitch            |
| 24    | lh_A_THJ5      | lh_THJ5      | Thumb base             |
| 25    | lh_A_THJ4      | lh_THJ4      | Thumb proximal         |
| 26    | lh_A_THJ3      | lh_THJ3      | Thumb hub              |
| 27    | lh_A_THJ2      | lh_THJ2      | Thumb middle           |
| 28    | lh_A_THJ1      | lh_THJ1      | Thumb distal           |
| 29    | lh_A_FFJ4      | lh_FFJ4      | Index finger knuckle   |
| 30    | lh_A_FFJ3      | lh_FFJ3      | Index finger proximal  |
| 31    | lh_A_FFJ0      | lh_FFJ0 (t)  | Index finger mid+dist  |
| 32    | lh_A_MFJ4      | lh_MFJ4      | Middle finger knuckle  |
| 33    | lh_A_MFJ3      | lh_MFJ3      | Middle finger proximal |
| 34    | lh_A_MFJ0      | lh_MFJ0 (t)  | Middle finger mid+dist |
| 35    | lh_A_RFJ4      | lh_RFJ4      | Ring finger knuckle    |
| 36    | lh_A_RFJ3      | lh_RFJ3      | Ring finger proximal   |
| 37    | lh_A_RFJ0      | lh_RFJ0 (t)  | Ring finger mid+dist   |
| 38    | lh_A_LFJ5      | lh_LFJ5      | Little finger metacarp |
| 39    | lh_A_LFJ4      | lh_LFJ4      | Little finger knuckle  |
| 40    | lh_A_LFJ3      | lh_LFJ3      | Little finger proximal |
| 41    | lh_A_LFJ0      | lh_LFJ0 (t)  | Little finger mid+dist |
| 42    | forearm_tx      | forearm_tx   | Forearm slide along keyboard |
| 43    | forearm_ty      | forearm_ty   | Forearm slide in/out (depth) |

> **Do not read the forearm DOF names as world axes.** `forearm_ty`'s MJCF axis
> is `(0, 0, 1)`, but that is the hand's *local* frame; measured in world
> coordinates it moves the fingertip along **x** (keyboard depth), with
> `|Δz| = 0.0000`. `forearm_tx` spans the keyboard (world **y**), also with
> `|Δz| = 0.0000`. **Neither forearm DOF has a vertical component** —
> `forearm_tz` is defined in `shadow_hand.py:_FOREARM_DOFS` but is absent from
> `_DEFAULT_FOREARM_DOFS`, so it is never instantiated. The only actuators that
> lower a fingertip onto a key are `WRJ1` (wrist pitch) and the five digits'
> flexion chains. Verified empirically (ClairDeLune, per-DOF fingertip
> displacement vs a zero-action baseline; forearm sweep with wrist+flexors
> frozen presses 0 keys).

## Sustain Pedal (Index 44)

| Index | Actuator | Description     |
|-------|----------|-----------------|
| 44    | sustain  | Sustain pedal   |

## Common Constraint Examples

```python
from omnipiano.safety.constraints import JointMagnitudeConstraint

# Limit right wrist pitch
JointMagnitudeConstraint(index=1, max_magnitude=0.8, penalty_coef=5.0)

# Limit left wrist pitch
JointMagnitudeConstraint(index=23, max_magnitude=0.8, penalty_coef=5.0)

# Limit both wrist yaw
JointMagnitudeConstraint(index=0, max_magnitude=0.5, penalty_coef=5.0)   # right
JointMagnitudeConstraint(index=22, max_magnitude=0.5, penalty_coef=5.0)  # left

# Limit right index finger knuckle
JointMagnitudeConstraint(index=7, max_magnitude=0.6, penalty_coef=3.0)
```

## Source Files

| File | Content |
|------|---------|
| `omnipiano/envs/robopianist/models/hands/third_party/shadow_hand/right_hand.xml` (L295-316) | Right hand actuator definitions in XML |
| `omnipiano/envs/robopianist/models/hands/third_party/shadow_hand/left_hand.xml` (L294-315) | Left hand actuator definitions in XML |
| `omnipiano/envs/robopianist/models/hands/shadow_hand_constants.py` (L21-31) | Joint groups and counts (NQ=24, NU=20) |
| `omnipiano/envs/robopianist/models/hands/shadow_hand.py` (L84-85) | Default forearm DOFs: `("forearm_tx", "forearm_ty")` |
| `omnipiano/envs/robopianist/suite/tasks/piano_with_shadow_hands.py` (L226-237) | Action spec: `merge([right_spec, left_spec, sustain_spec])` |
| `omnipiano/envs/robopianist/suite/tasks/piano_with_shadow_hands_test.py` (L96-117) | Test confirming action layout |

## Notes

- Actions are normalized to [-1.0, 1.0] by `RescaleAction` wrapper before reaching constraints.
  So `max_magnitude` in `JointMagnitudeConstraint` should be between 0.0 and 1.0.
- If `reduced_action_space=True`, actuators THJ5, THJ1, LFJ5 are removed per hand,
  which shifts all subsequent indices. The default configuration does NOT use reduced action space.
- Forearm DOFs are dynamically added by `ShadowHand._add_dofs()` and appended after the 20 XML actuators.
