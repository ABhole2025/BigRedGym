from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg


class Go2HeightCfg(Go2Cfg):
    class init_state(Go2Cfg.init_state):
        reset_mode = "reset_to_basic"

    class commands(Go2Cfg.commands):
        resampling_time = 2.0

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = 0.0
            yaw_vel = 0.0


class Go2HeightRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "height_command",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]

    class critic(Go2RunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "height_command",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
        ]

        class reward(Go2RunnerCfg.critic.reward):
            class weights(Go2RunnerCfg.critic.reward.weights):
                tracking_base_height = 4.0
                tracking_lin_vel = 0.0
                tracking_ang_vel = 0.0
                orientation = 1.0
                action_rate = 0.1
                action_rate2 = 0.01
                min_base_height = 0.0

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2height"
        max_iterations = 500
