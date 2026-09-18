# Overview

OmniPiano is a unified benchmark for learning dexterous multi-hand piano
playing. Built on RoboPianist and the MuJoCo physics engine, it turns piano
performance into a shared testbed for standard, robust, safe, and cooperative
multi-agent reinforcement learning.



## 🧠Why piano playing?

Dexterous piano playing provides a suitable testbed by combining precise
control with measurable task outcomes:

- **Precise control.** The task requires coordinated finger movements to press,
  hold, and release keys at the times specified by a MIDI score.
- **Measurable task outcomes.** These score-defined targets allow execution
  accuracy to be measured by comparing the intended and actual key activations.
- **Systematic difficulty variation.** Control difficulty can be varied through
  musical complexity, the number and workspaces of the hands, and fingering
  freedom.

## 🔥Benchmark tracks

All four tracks share the same piano, MIDI objective, and evaluation basis, and
each varies one aspect of the task — so a performance difference can be
attributed to that single design choice.

- **Standard RL.** Scales repertoire and hand morphology: up to 150 pieces and one
  to five Shadow Hands, with fully mobile hands or static partitions that clamp
  each hand to a register. The reward combines key activation, fingertip-to-key
  matching, and actuation regularization, under annotation-based or
  optimal-transport (OT) fingering.
- **Robust RL.** Perturbs the agent–environment interaction at six targets —
  action, observation, reward, gravity, fingertip–key friction, and initial
  hand pose — each under Gaussian, bounded-uniform, or constant-shift noise.
  Channels compose into compound stress tests.
- **Safe RL.** Keeps musical reward and physical safety cost as separate signals,
  covering joint magnitude, inter-hand collision, actuator power, and
  joint-injury risk. Feasibility is judged before musical quality, and a policy
  that stays inactive to keep its cost low counts as a trivial-safe failure.
- **Multi-agent RL.** Distributes a fixed set of hands across decentralized agents
  that share one musical objective, exposed through a PettingZoo
  `ParallelEnv`. Cooperation difficulty is organized along scalability,
  coupling, heterogeneity, and observability (**SCHO**).

## 🔥Benchmark Features

- **Unified task family**: Standard, robust, safe, and multi-agent RL share one
  piano, one MIDI objective, and one evaluation basis, each varying a single
  aspect of the same task.
- **Scalable morphology**: One to five Shadow Hands — up to 111 continuous
  action dimensions — with optional static partitions that clamp each hand to a
  keyboard register.
- **Comprehensive task coverage**: Up to 150 pieces across eight hand settings,
  exposed as 600+ registered task IDs spanning all four tracks.
- **High compatibility**: Gymnasium and PettingZoo APIs with vectorized
  environment support, usable from Stable-Baselines3, RLlib, CleanRL, OmniSafe,
  and TorchRL.

## 📷Demonstrations

### Multi-hand morphology

```{figure} ../../demos/morphology/5hand_winterwind_l1.gif
:alt: Five-hand Winter Wind static-partition demonstration
:width: 90%
:align: center

Five-hand *Winter Wind* with a Level-1 static keyboard partition. Each hand is
assigned a register, encouraging all five hands to participate in the piece.
```

Static partitions provide a controlled comparison with unrestricted
morphologies: the music and hand geometry can remain fixed while the permitted
keyboard workspace changes.

### Decentralized cooperation

```{figure} ../../demos/morphology/4hand_winterwind_ma_duet.gif
:alt: Four-hand multi-agent Winter Wind duet
:width: 90%
:align: center

Four-hand *Winter Wind* performed as a cooperative duet by two decentralized
agents controlling the bass-side and treble-side hand pairs.
```

### Explicit safety constraints

```{figure} ../../demos/Collision_Safe/preview.gif
:alt: Collision-aware piano-playing task
:width: 75%
:align: center

Collision-aware piano playing. Musical reward and collision cost are exposed
as separate signals so safe-RL algorithms can optimize their trade-off.
```

Continue with the [Quick Start](quick_start.md) to install OmniPiano and run a
first environment.
