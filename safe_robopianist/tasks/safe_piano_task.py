from robopianist.suite.tasks import piano_with_shadow_hands
from dm_control import mjcf
from safe_robopianist.configs import TaskVariantConfig

class SafePianoTask(piano_with_shadow_hands.PianoWithShadowHands):
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
        root_mjcf = self.root_entity.mjcf_model
        
        # Example: Make the left hand immobile by removing its actuators
        if self.task_config.left_hand_immobile:
            # Find all actuators in the XML tree
            all_actuators = root_mjcf.find_all('actuator')
            for actuator in all_actuators:
                # If the actuator belongs to the left hand, remove it
                if "left_" in actuator.name:
                    actuator.remove()
            
            # Optionally, we can also make the left hand joints fixed
            # so it doesn't flop around due to gravity
            all_joints = root_mjcf.find_all('joint')
            for joint in all_joints:
                if "left_" in joint.name:
                    joint.type = "fixed"
                    
        # Example: Make the right hand immobile
        if self.task_config.right_hand_immobile:
            all_actuators = root_mjcf.find_all('actuator')
            for actuator in all_actuators:
                if "right_" in actuator.name:
                    actuator.remove()
            all_joints = root_mjcf.find_all('joint')
            for joint in all_joints:
                if "right_" in joint.name:
                    joint.type = "fixed"

    # We can also override before_step to implement Dynamics Randomization
    def before_step(self, physics, action, random_state):
        # TODO: Implement domain randomization (e.g., changing friction) here
        super().before_step(physics, action, random_state)
