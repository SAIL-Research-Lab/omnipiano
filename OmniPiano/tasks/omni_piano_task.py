from robopianist.suite.tasks import piano_with_shadow_hands
from dm_control import mjcf
from OmniPiano.configs import TaskVariantConfig


class OmniPianoTask(piano_with_shadow_hands.PianoWithShadowHands):
    """
    A custom Task that intercepts the environment compilation process.
    This allows us to dynamically modify the XML (mjcf_model) in memory
    before the physics engine is initialized.
    """
    def __init__(self, task_config: TaskVariantConfig, **kwargs):
        self.task_config = task_config

        if task_config.disable_fingering_reward:
            kwargs["disable_fingering_reward"] = True

        # 1. Call the parent constructor.
        # This builds the default XML tree (mjcf_model) in memory.
        super().__init__(**kwargs)
        
        # 2. Dynamically modify the XML tree based on the config
        self._apply_task_variants()
        
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

    # We can also override before_step to implement Dynamics Randomization
    def before_step(self, physics, action, random_state):
        # TODO: Implement domain randomization (e.g., changing friction) here
        super().before_step(physics, action, random_state)
