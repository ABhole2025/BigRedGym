
from go2_deploy.custom_implementation.custom_config import CustomConfig
from gym.envs.go2.go2boxstep_config import Go2BoxStepCfg


class Go2BoxStepDeploy(CustomConfig):
    task_name = "go2boxstep"

    # Must match Go2BoxStepRunnerCfg.actor.obs
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "beat_clock_obs",
        "tempo_bpm",
        "beat_of_box_obs",
        "step_size",
    ]

    custom_obs_sizes = {
        "beat_clock_obs": 2,
        "tempo_bpm": 1,
        "beat_of_box_obs": 6,
        "step_size": 1,
    }

    # Training uses zero default joint angles.
    # The gait reference supplies the actual posture.
    default_dof_pos = 12 * [0.0]

    class DeployScaling(Go2BoxStepCfg.scaling):
        pass

    def get_custom_buffers_dict(self):
        return {}

    def update_custom_buffers(self, main_controller, t):
        pass

    def get_custom_obs_pieces(self):
        return {}
