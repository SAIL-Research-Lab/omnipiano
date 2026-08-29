import numpy as np

from robopianist.suite.tasks import piano_with_shadow_hands
from omnipiano.configs import RobustConfig, TaskVariantConfig


ENVIRONMENT_NOISE_SEED_OFFSET = 30000


class OmniPianoTask(piano_with_shadow_hands.PianoWithShadowHands):
    """
    A custom Task that intercepts the environment compilation process.
    This allows us to dynamically modify the XML (mjcf_model) in memory
    before the physics engine is initialized.
    """
    _ENV_NOISE_Z_OFFSET_LIMIT = 0.05
    _MIN_FRICTION = 1e-6
    _MAX_GRAVITY_Z = -1e-6

    def __init__(
        self,
        task_config: TaskVariantConfig,
        robust_config: RobustConfig = None,
        environment_noise_seed=None,
        **kwargs,
    ):
        self.task_config = task_config
        self.robust_config = robust_config or RobustConfig()
        self._environment_noise_rng = np.random.default_rng(
            environment_noise_seed
        )
        self._nominal_gravity = None
        self._nominal_tip_friction = None
        self._nominal_key_friction = None
        self._nominal_hand_poses = None
        self._environment_noise_state = self._zero_environment_noise_state()
        self._environment_noise_active = any(
            self.robust_config.is_channel_active(parameter)
            for parameter in self.robust_config.environment_noise.PARAMETERS
        )

        # 1. Call the parent constructor.
        # This builds the default XML tree (mjcf_model) in memory.
        super().__init__(**kwargs)

        # 2. Dynamically modify the XML tree based on the config
        self._apply_task_variants()

    def _zero_environment_noise_state(self, physics=None):
        gravity_z = 0.0
        contact_friction_sliding = 0.0
        hand_position_offsets = {}
        if physics is not None:
            gravity_z = float(physics.model.opt.gravity[2])
            contact_friction_sliding = float(np.mean(
                physics.bind(self.piano.key_geoms).friction[:, 0]
            ))
            hand_position_offsets = {
                spec.name: {"y": 0.0, "z": 0.0}
                for spec in self.hand_specs
            }
        return {
            "gravity_noise": 0.0,
            "gravity_z": gravity_z,
            "contact_friction_noise": 0.0,
            "contact_friction_sliding": contact_friction_sliding,
            "hand_position_offsets": hand_position_offsets,
            "hand_position_l2": 0.0,
            "hand_position_max_l2": 0.0,
        }

    @property
    def environment_noise_state(self):
        """Copy of the physical perturbations active in this episode."""
        state = dict(self._environment_noise_state)
        state["hand_position_offsets"] = {
            name: dict(offset)
            for name, offset in state["hand_position_offsets"].items()
        }
        return state
        
    # Move frozen hands far below the piano so they don't interfere.
    _FROZEN_HAND_POSITION = (0.4, 0.0, -0.5)

    def _apply_task_variants(self):
        if self.task_config.left_hand_immobile:
            self._freeze_hand(self.left_hand)
        if self.task_config.right_hand_immobile:
            self._freeze_hand(self.right_hand)

    def _freeze_hand(self, hand):
        """Lock all joints/actuators and relocate the hand below the piano.

        We cannot remove actuators or joints because MJCF sensors (actuatorvel,
        actuatorfrc) and tendons hold live references to them. Removing any
        element would leave dangling references and crash XML compilation.
        Instead we clamp both joint ranges and actuator control ranges to near-zero,
        then move the hand's root body far below the stage so it cannot physically
        interfere with the active hand or the piano keys.
        """
        # MuJoCo requires range[0] < range[1], so use a tiny epsilon interval
        # instead of [0, 0] to approximate a fully locked joint/actuator.
        _EPS = 1e-7
        for joint in list(hand.joints):
            # `joint.range` is only enforced when `joint.limited` is enabled.
            joint.limited = True
            joint.range = [-_EPS, _EPS]
        for actuator in list(hand.actuators):
            # Clamp controller outputs to near zero so policy commands cannot move this hand.
            actuator.ctrlrange = [-_EPS, _EPS]
        # Relocate the entire hand away from the keyboard to eliminate residual contacts.
        hand.root_body.pos = self._FROZEN_HAND_POSITION

    def disable_rendering_bookkeeping(self):

        self._disable_colorization = True
        self.piano.disable_rendering_bookkeeping()

    def initialize_episode(self, physics, random_state):
        """Reset task state, then sample one stationary physical model."""
        self._restore_nominal_model_parameters(physics)
        super().initialize_episode(physics, random_state)
        if self._environment_noise_active:
            self._apply_environment_noise(physics)
        else:
            self._environment_noise_state = self._zero_environment_noise_state(
                physics
            )

    def _cache_nominal_environment(self, physics):
        if self._nominal_gravity is not None:
            return
        self._nominal_gravity = physics.model.opt.gravity.copy()
        tip_geoms = [
            geom
            for hand in self.hands
            for geom in hand.fingertip_collision_geoms
        ]
        self._tip_collision_geoms = tuple(tip_geoms)
        self._key_collision_geoms = tuple(self.piano.key_geoms)
        self._nominal_tip_friction = (
            physics.bind(self._tip_collision_geoms).friction.copy()
        )
        self._nominal_key_friction = (
            physics.bind(self._key_collision_geoms).friction.copy()
        )
        # Entity.shift_pose() changes the compiled attachment-frame pose, which
        # is a model parameter and therefore survives composer resets. Cache
        # every hand's pristine pose so hand-position domain randomization is
        # episode-local rather than an accumulating random walk.
        self._nominal_hand_poses = tuple(
            tuple(
                np.asarray(value, dtype=float).copy()
                for value in hand.get_pose(physics)
            )
            for hand in self.hands
        )

    def _restore_nominal_model_parameters(self, physics):
        if self._nominal_gravity is None:
            return
        physics.model.opt.gravity[:] = self._nominal_gravity
        physics.bind(self._tip_collision_geoms).friction = (
            self._nominal_tip_friction
        )
        physics.bind(self._key_collision_geoms).friction = (
            self._nominal_key_friction
        )

    def _apply_environment_noise(self, physics):
        self._cache_nominal_environment(physics)

        # Model parameters persist across composer resets, so always restore
        # the compiled nominal values before applying the next episode draw.
        self._restore_nominal_model_parameters(physics)
        for hand, (position, quaternion) in zip(
            self.hands, self._nominal_hand_poses
        ):
            hand.set_pose(
                physics,
                position=position.copy(),
                quaternion=quaternion.copy(),
            )
        # xpos is derived data. Refresh it before hand-position clipping reads
        # the restored world-space position below.
        physics.forward()

        gravity_noise, friction_noise = self._sample_and_apply_model_noise(
            physics, frequency="episode"
        )

        offsets = self._sample_and_apply_hand_position_noise(physics)
        offset_vectors = np.asarray(
            [[v["y"], v["z"]] for v in offsets.values()], dtype=float
        )
        per_hand_l2 = (
            np.linalg.norm(offset_vectors, axis=1)
            if offset_vectors.size
            else np.zeros(0, dtype=float)
        )

        physics.forward()
        self._environment_noise_state = {
            "gravity_noise": (
                gravity_noise if gravity_noise is not None else 0.0
            ),
            "gravity_z": float(physics.model.opt.gravity[2]),
            "contact_friction_noise": (
                friction_noise if friction_noise is not None else 0.0
            ),
            "contact_friction_sliding": float(
                np.mean(physics.bind(self._key_collision_geoms).friction[:, 0])
            ),
            "hand_position_offsets": offsets,
            "hand_position_l2": float(np.linalg.norm(offset_vectors)),
            "hand_position_max_l2": float(np.max(per_hand_l2))
            if per_hand_l2.size else 0.0,
        }

    def _sample_and_apply_model_noise(self, physics, frequency):
        """Apply gravity then friction draws for one sampling frequency.

        Every value is derived from the cached nominal model, so step-level
        noise cannot accumulate into a random walk. This method intentionally
        does not call ``physics.forward()``: gravity and geom friction are read
        by the subsequent physics integration without refreshing derived data.
        """
        env_config = self.robust_config.environment_noise

        gravity_noise = None
        if (
            env_config.gravity_frequency == frequency
            and self.robust_config.is_channel_active("gravity")
        ):
            requested = float(self.robust_config.sample_noise(
                self._environment_noise_rng, "gravity"
            ))
            nominal_z = float(self._nominal_gravity[2])
            applied_z = min(nominal_z + requested, self._MAX_GRAVITY_Z)
            gravity_noise = applied_z - nominal_z
            physics.model.opt.gravity[2] = applied_z

        friction_noise = None
        if (
            env_config.contact_friction_frequency == frequency
            and self.robust_config.is_channel_active("contact_friction")
        ):
            requested = float(self.robust_config.sample_noise(
                self._environment_noise_rng, "contact_friction"
            ))
            tip_friction = self._nominal_tip_friction.copy()
            key_friction = self._nominal_key_friction.copy()
            tip_friction[:, 0] = np.maximum(
                tip_friction[:, 0] + requested, self._MIN_FRICTION
            )
            key_friction[:, 0] = np.maximum(
                key_friction[:, 0] + requested, self._MIN_FRICTION
            )
            physics.bind(self._tip_collision_geoms).friction = tip_friction
            physics.bind(self._key_collision_geoms).friction = key_friction
            friction_noise = float(
                np.mean(key_friction[:, 0] - self._nominal_key_friction[:, 0])
            )

        return gravity_noise, friction_noise

    def _apply_step_environment_noise(self, physics):
        gravity_noise, friction_noise = self._sample_and_apply_model_noise(
            physics, frequency="step"
        )
        if gravity_noise is not None:
            self._environment_noise_state["gravity_noise"] = gravity_noise
            self._environment_noise_state["gravity_z"] = float(
                physics.model.opt.gravity[2]
            )
        if friction_noise is not None:
            self._environment_noise_state[
                "contact_friction_noise"
            ] = friction_noise
            self._environment_noise_state[
                "contact_friction_sliding"
            ] = float(np.mean(
                physics.bind(self._key_collision_geoms).friction[:, 0]
            ))

    def _sample_and_apply_hand_position_noise(self, physics):
        n_hands = len(self.hands)
        dy = np.zeros(n_hands, dtype=float)
        dz = np.zeros(n_hands, dtype=float)
        if self.robust_config.is_channel_active("hand_position_y"):
            dy = np.asarray(self.robust_config.sample_noise(
                self._environment_noise_rng,
                "hand_position_y",
                shape=(n_hands,),
            ), dtype=float)
        if self.robust_config.is_channel_active("hand_position_z"):
            dz = np.asarray(self.robust_config.sample_noise(
                self._environment_noise_rng,
                "hand_position_z",
                shape=(n_hands,),
            ), dtype=float)

        offsets = {}
        piano_y_limit = float(self.piano.size[1])
        for index, (hand, spec) in enumerate(zip(self.hands, self.hand_specs)):
            current_y = float(physics.bind(hand.root_body).xpos[1])
            y_range = spec.resolved_y_range
            if y_range is None:
                y_range = (-piano_y_limit, piano_y_limit)
            target_y = float(np.clip(current_y + dy[index], *y_range))
            applied_y = target_y - current_y
            applied_z = float(np.clip(
                dz[index],
                -self._ENV_NOISE_Z_OFFSET_LIMIT,
                self._ENV_NOISE_Z_OFFSET_LIMIT,
            ))
            if applied_y != 0.0 or applied_z != 0.0:
                hand.shift_pose(physics, (0.0, applied_y, applied_z))
            offsets[spec.name] = {"y": applied_y, "z": applied_z}
        return offsets

    def before_step(self, physics, action, random_state):
        if self._environment_noise_active:
            self._apply_step_environment_noise(physics)
        super().before_step(physics, action, random_state)
