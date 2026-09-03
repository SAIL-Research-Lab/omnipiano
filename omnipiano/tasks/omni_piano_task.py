from robopianist.suite.tasks import piano_with_shadow_hands
from omnipiano.configs import TaskVariantConfig


class OmniPianoTask(piano_with_shadow_hands.PianoWithShadowHands):
    """
    A custom Task that intercepts the environment compilation process.
    This allows us to dynamically modify the XML (mjcf_model) in memory
    before the physics engine is initialized.
    """
    def __init__(self, task_config: TaskVariantConfig, **kwargs):
        self.task_config = task_config

        # 1. Call the parent constructor.
        # This builds the default XML tree (mjcf_model) in memory.
        super().__init__(**kwargs)

        # 2. Dynamically modify the XML tree based on the config
        self._apply_task_variants()
        
    def _apply_task_variants(self):
        if self.task_config.left_hand_immobile:
            self.left_hand.detach()
        if self.task_config.right_hand_immobile:
            self.right_hand.detach()

    # We can also override before_step to implement Dynamics Randomization
    def before_step(self, physics, action, random_state):
        # TODO: Implement domain randomization (e.g., changing friction) here
        super().before_step(physics, action, random_state)
