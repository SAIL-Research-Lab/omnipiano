# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base piano composer task."""

from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils

from OmniPiano.tasks.hand_spec import HandSpec, default_two_hand_specs
from robopianist.models.hands import HandSide, shadow_hand
from robopianist.models.piano import piano

# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.

# Default position and orientation of the hands.
_LEFT_HAND_POSITION = (0.4, -0.15, 0.13)
_LEFT_HAND_QUATERNION = (-1, -1, 1, 1)
_RIGHT_HAND_POSITION = (0.4, 0.15, 0.13)
_RIGHT_HAND_QUATERNION = (-1, -1, 1, 1)

_ATTACHMENT_YAW = 0  # Degrees.


class PianoOnlyTask(composer.Task):
    """Piano task with no hands."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        change_color_on_activation: bool = False,
        add_piano_actuators: bool = False,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self._piano = piano.Piano(
            change_color_on_activation=change_color_on_activation,
            add_actuators=add_piano_actuators,
        )
        arena.attach(self._piano)

        # Harden the piano keys.
        # The default solref parameters are (0.02, 1). In particular, the first
        # parameter specifies -stiffness, and so decreasing it makes the contacts
        # harder. The documentation recommends keeping the stiffness at least 2x larger
        # than the physics timestep, see:
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=stiffness#solver-parameters
        self._piano.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    @property
    def piano(self) -> piano.Piano:
        return self._piano

    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0


class PianoTask(PianoOnlyTask):
    """Base class for piano tasks."""

    def __init__(
        self,
        arena: composer_utils.Arena,
        gravity_compensation: bool = False,
        change_color_on_activation: bool = False,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        attachment_yaw: float = _ATTACHMENT_YAW,
        forearm_dofs: Sequence[str] = shadow_hand._DEFAULT_FOREARM_DOFS,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
        hand_specs: Optional[Sequence[HandSpec]] = None,
    ) -> None:
        super().__init__(
            arena=arena,
            change_color_on_activation=change_color_on_activation,
            add_piano_actuators=False,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )

        if hand_specs is None:
            # Legacy 2-hand path: build the canonical ("rh", "lh") pair from
            # default_two_hand_specs(), and thread the legacy kwargs
            # (attachment_yaw, forearm_dofs, reduced_action_space) into each
            # spec so existing tasks get the exact behavior they had before.
            base_specs = default_two_hand_specs()
            hand_specs = tuple(
                HandSpec(
                    name=s.name,
                    side=s.side,
                    position=s.position,
                    quaternion=s.quaternion,
                    attachment_yaw=attachment_yaw,
                    forearm_dofs=tuple(forearm_dofs),
                    reduced_action_space=reduced_action_space,
                    group=s.group,
                )
                for s in base_specs
            )
        else:
            hand_specs = tuple(hand_specs)

        names = [s.name for s in hand_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"HandSpec names must be unique, got: {names}")

        self._hand_specs: List[HandSpec] = list(hand_specs)
        self._hands: List[shadow_hand.ShadowHand] = []
        self._hands_by_name: Dict[str, shadow_hand.ShadowHand] = {}
        for spec in self._hand_specs:
            hand = self._add_hand(
                spec=spec,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
            )
            self._hands.append(hand)
            self._hands_by_name[spec.name] = hand

    # Accessors.

    @property
    def hands(self) -> List[shadow_hand.ShadowHand]:
        """All hands attached to this task, in spec order."""
        return list(self._hands)

    @property
    def hand_specs(self) -> List[HandSpec]:
        return list(self._hand_specs)

    @property
    def hands_by_name(self) -> Dict[str, shadow_hand.ShadowHand]:
        return dict(self._hands_by_name)

    def hands_in_group(self, group: str) -> List[shadow_hand.ShadowHand]:
        """Return all hands whose spec.group matches `group` (for multi-agent partitioning)."""
        return [
            hand
            for hand, spec in zip(self._hands, self._hand_specs)
            if spec.group == group
        ]

    @property
    def left_hand(self) -> shadow_hand.ShadowHand:
        """Backward-compat accessor: the first hand with `HandSide.LEFT`.

        (If a caller explicitly registered a spec named "left", that takes
        priority; default 2-hand specs use names "rh"/"lh", so the side
        fallback is what normally resolves.)
        """
        return self._canonical_hand("left", HandSide.LEFT)

    @property
    def right_hand(self) -> shadow_hand.ShadowHand:
        """Backward-compat accessor: the first hand with `HandSide.RIGHT`.

        (If a caller explicitly registered a spec named "right", that takes
        priority; default 2-hand specs use names "rh"/"lh", so the side
        fallback is what normally resolves.)
        """
        return self._canonical_hand("right", HandSide.RIGHT)

    def _canonical_hand(
        self, name: str, side: HandSide
    ) -> shadow_hand.ShadowHand:
        if name in self._hands_by_name:
            return self._hands_by_name[name]
        for hand, spec in zip(self._hands, self._hand_specs):
            if spec.side == side:
                return hand
        raise AttributeError(
            f"No hand with name {name!r} or side {side} in this task."
        )

    # Helper methods.

    def _add_hand(
        self,
        spec: HandSpec,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
    ) -> shadow_hand.ShadowHand:
        # Static partition (Level 1+): if the spec carries y_range or
        # key_range, override forearm_tx joint range with the resolved arena
        # Y bounds. Otherwise fall back to the original "full keyboard
        # reach" behavior — preserving 2-hand and Level-3 N-hand semantics.
        # ``HandSpec.resolved_y_range`` is the single source of truth; do
        # NOT inspect spec.key_range / spec.y_range directly here.
        partition_y_range = spec.resolved_y_range
        if partition_y_range is not None:
            y_lo, y_hi = partition_y_range
            joint_range = [y_lo - spec.position[1], y_hi - spec.position[1]]
        else:
            joint_range = [-self._piano.size[1], self._piano.size[1]]
            # Offset the joint range by the hand's initial position.
            joint_range[0] -= spec.position[1]
            joint_range[1] -= spec.position[1]

        hand = shadow_hand.ShadowHand(
            name=f"{spec.name}_shadow_hand",
            side=spec.side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=spec.reduced_action_space,
            forearm_dofs=spec.forearm_dofs,
        )
        hand.root_body.pos = spec.position

        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        # LEFT-sided hands get the opposite yaw sign to stay symmetric.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if spec.side == HandSide.LEFT else 1
        angle = np.radians(sign * spec.attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, spec.quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand
