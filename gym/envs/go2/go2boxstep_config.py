# AI-assisted: box-step task and runner configs built from BOXSTEP_PLAN.md and the
# config section of BOXSTEP_PROJECT_MAP.md. Reward weights and the velocity
# tracking width are provisional starting values, not decisions by the user.
# Built on Go2Cfg since 2026-11-20 (PR review); the values that used to come
# from Go2TrotCfg are now set here explicitly.
from gym.envs.go2.go2_config import BASE_HEIGHT_REF, Go2Cfg, Go2RunnerCfg

STANDING_HEIGHT_M = 0.30  # hip joint above the foot centres, decided 2026-11-10
FOOT_RADIUS_M = 0.022  # go2.urdf foot collision sphere


class Go2BoxStepCfg(Go2Cfg):
    class env(Go2Cfg.env):
        episode_length_s = 12  # about 2.3 boxes at 70 BPM (one box = 5.1 s)

    class init_state(Go2Cfg.init_state):
        # The reference supplies absolute joint angles, so the default posture
        # the policy's corrections are added to is zero.
        default_joint_angles = {"hip": 0.0, "thigh": 0.0, "calf": 0.0}
        # Robots reset in the standing pose (leg IK of the neutral feet). The hip
        # joints sit level with the base origin in go2.urdf, so the base starts
        # at the standing height plus the foot radius, feet touching the floor.
        pos = [0.0, 0.0, STANDING_HEIGHT_M + FOOT_RADIUS_M]
        reset_mode = "reset_to_basic"

    class boxstep:
        tempo_bpm_range = [70.0, 70.0]  # changed from 105 BPM on 2026-11-13
        step_size_range = [0.25, 0.30]  # d [m]
        # sigma: one leg pair's swing as a fraction of the beat, so it scales
        # with the tempo (PR review). 0.2 s at 70 BPM, as before.
        swing_fraction = 0.2 * 70.0 / 60.0
        standing_height_m = STANDING_HEIGHT_M  # h
        swing_clearance_m = 0.06  # g_c
        # Provisional starting value (not decided by the user): width of the
        # exp(-|v_ref - v|^2 / width^2) body-velocity reward [m/s]. Reference
        # peak speed is about 1 m/s at d = 0.25 m, so 0.3 m/s is tighter than
        # go2trot's tracking_sigma = 0.25 (an effective width of 0.5 m/s).
        velocity_tracking_width = 0.3
        # Provisional: width of the exp(-(omega_z / width)^2) yaw-rate reward
        # [rad/s]. At 0.3, the 0.28 rad/s of run Nov13_02-15-31_ scores 0.42.
        yaw_rate_tracking_width = 0.3
        # Provisional: width of the exp(-|Δx - d·p·e|^2 / width^2) per-beat
        # displacement reward [m]. Half a 0.27 m step short scores about 0.16.
        displacement_tracking_width = 0.10
        # Provisional: width of the exp(-err^2 / width^2) stance-foot placement
        # reward [m]. Run Nov13_15-31-54_ had a 16 cm median error (scores 0.26
        # mean); the reference alone has 9 cm from joint lag (scores 0.42).
        foot_placement_width = 0.10

    class asset(Go2Cfg.asset):
        # Head_upper and Head_lower are separate bodies from base on the Go2.
        # Run Nov10_21-15-58_ learned to rest its face on the floor at 40 deg
        # pitch because only base contact ended an episode.
        terminate_after_contacts_on = ["base", "Head"]

    class push_robots(Go2Cfg.push_robots):
        toggle = False  # no pushes until the box works
        interval_s = 5

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            link_mass_scale_range = [0.9, 1.2]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {
                "p_gains": [0.9, 1.1],
                "d_gains": [0.8, 1.2],
            }

    class reward_settings(Go2Cfg.reward_settings):
        # Provisional starting value: vertical foot force that counts as contact
        # for touchdown counting, support reward and slip penalty [N]
        # (about 3 % of body weight).
        foot_contact_force_n = 5.0
        # Base height the min_base_height reward saturates at [m]: the reset
        # height. The reference-only run sagged to about 0.20 m with nothing
        # rewarding the body for staying up.
        base_height_target = STANDING_HEIGHT_M + FOOT_RADIUS_M

    class scaling(Go2Cfg.scaling):
        # Same scale as base_lin_vel so the critic sees both in the same units.
        reference_body_velocity = BASE_HEIGHT_REF
        beat_displacement = 0.3  # about one step [m]
        tempo_bpm = 100.0  # 70 BPM observed as 0.7


class Go2BoxStepRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        # 52 numbers. No velocity commands, no foot contacts. Timing inputs
        # (slimmed after PR review): the music clock (sin, cos of the beat),
        # the tempo, and which beat of the box it is (one-hot, 6).
        obs = [
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
        smooth_exploration = False

    class critic(Go2RunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "beat_clock_obs",
            "tempo_bpm",
            "beat_of_box_obs",
            "step_size",
            "reference_body_velocity",
            "beat_displacement",
        ]
        normalize_obs = False

        class reward(Go2RunnerCfg.critic.reward):
            # Provisional starting values, modeled on Go2TrotRunnerCfg. None of
            # these weights has been decided by the user; review before training.
            class weights(Go2RunnerCfg.critic.reward.weights):
                # Off: no velocity commands in this task.
                tracking_lin_vel = 0.0
                # Kept from go2trot (same values unless noted).
                tracking_ang_vel = 2.0  # yaw rate target is zero
                lin_vel_z = 0.25  # go2trot has 0.0; the plan keeps this term
                ang_vel_xy = 0.01
                # Raised from go2trot's 1.0: at 1.0, 40 deg of pitch cost only
                # 0.8 and the policy traded it for velocity reward.
                orientation = 3.0
                torques = 5.0e-6
                # Hold the body at the standing height (go2trot has 0.0 here and
                # uses the height command instead; 1.5 is Go2RunnerCfg's value).
                min_base_height = 1.5
                action_rate = 0.25
                action_rate2 = 0.025
                # Pace schedule (same values as trot_support / swing_contact).
                # Raised from 0.625: run Nov13_01-15-36_ kneeled on its shins with
                # its feet down under 10% of the time.
                scheduled_support = 2.0
                # Calf (shin/knee) ground contact, one point per calf body
                # touching (asset.penalize_contacts_on). Was unset, so 0.
                collision = 1.0
                swing_contact = 1.25
                # New box-step terms.
                tracking_body_velocity = 4.0  # as go2trot tracking_lin_vel
                # Body reaches each corner (run Nov13_02-32-57_ covered 19-67 %
                # of each step). Provisional.
                tracking_beat_displacement = 4.0
                # Stance feet stay under their hips as the reference places them
                # (back-right foot dragged 20 cm behind in Nov13_15-31-54_).
                stance_foot_placement = 2.0
                stance_foot_slip = 0.5
                # Raised from 0.5: run Nov13_18-25-51_ hopped after each step, all
                # four feet off the ground 15 % of the time they should be down.
                extra_touchdowns = 2.0
                # Raised from 0.1: run Nov13_01-51-32_ overrode the reference by
                # over 1 rad per joint and crouched with one leg tucked up.
                correction_size = 2.0

    class algorithm(Go2RunnerCfg.algorithm):
        # Same PPO settings as Go2TrotRunnerCfg.
        gamma = 0.99
        lam = 0.95
        batch_size = 2**15
        max_gradient_steps = 24
        clip_param = 0.2
        learning_rate = 1.0e-3
        max_grad_norm = 1.0
        rollout_size = 2**16
        use_clipped_value_loss = True
        entropy_coef = 0.01
        schedule = "adaptive"
        desired_kl = 0.01
        lr_range = [2e-5, 1e-2]
        lr_ratio = 1.5

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2boxstep"
        max_iterations = 1000  # what our best runs used
