OmniPiano Tutorial
==================

Achieving human-like dexterity in robotic hands remains a major challenge in robotics. Piano playing exemplifies this challenge, requiring precise spatial and temporal coordination of multiple fingers and arms in a high-dimensional environment, and provides a demanding testbed for reinforcement learning (RL). Although RoboPianist established a benchmark for robotic piano playing, it is limited to two hands and lacks a unified evaluation framework for robust, safe, and multi-agent reinforcement learning. To address this gap, we introduce OmniPiano, a benchmark that extends RoboPianist from two to up to five Shadow Hands and supports standard, robust, safe, and multi-agent RL within a shared piano-playing task family. OmniPiano combines scalable multi-hand control with configurable perturbations, explicit safety constraints, and decentralized cooperation settings. We additionally provide an open-source implementation, high modularity design, comprehensive task coverage to support RL study of task performance, robustness, safety, and cooperation in dexterous control.

.. figure:: _static/images/framework_v1.png
   :alt: OmniPiano architecture
   :width: 90%
   :align: center
   :figclass: bold-italic-caption

   The overview of OmniPiano Benchmark. A unified benchmark for learning dexterous multi-hand piano playing.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   introduction/overview
   introduction/quick_start
