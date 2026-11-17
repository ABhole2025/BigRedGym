# AI-assisted: box-step task and runner configs built from BOXSTEP_PLAN.md and the
# config section of BOXSTEP_PROJECT_MAP.md. Reward weights and the velocity
# tracking width are provisional starting values, not decisions by the user.
from gym.envs.go2.go2_config import BASE_HEIGHT_REF
from gym.envs.go2.go2trot_config import Go2TrotCfg, Go2TrotRunnerCfg

STANDING_HEIGHT_M = 0.30  # hip joint above the foot centres, decided 2026-11-10
FOOT_RADIUS_M = 0.022  # go2.urdf foot collision sphere


class Go2BoxStepCfg(Go2TrotCfg):
    class env(Go2TrotCfg.env):
        episode_length_s = 21  # about two boxes at 70 BPM (12 beats = 10.3 s)

    class init_state(Go2TrotCfg.init_state):
        # Robots reset in the standing pose (leg IK of the neutral feet). The hip
        # joints sit level with the base origin in go2.urdf, so the base starts
        # at the standing height plus the foot radius, feet touching the floor.
        pos = [0.0, 0.0, STANDING_HEIGHT_M + FOOT_RADIUS_M]
        reset_mode = "reset_to_basic"

    class boxstep:
        tempo_bpm_range = [70.0, 70.0]  # changed from 105 BPM on 2026-11-13
        # d [m]: how far each side's feet move per bar. 18 cm, decided
        # 2026-11-14: one side stepping alone cannot reach 25-30 cm sideways.
        step_size_range = [0.18, 0.18]
        swing_time_s = 0.2  # T_sw, one leg pair's swing
        standing_height_m = STANDING_HEIGHT_M  # h
        swing_clearance_m = 0.06  # g_c
        # Provisional: width of the exp(-|v_ref - v|^2 / width^2) body-velocity
        # reward [m/s]. Peak reference speed is about 0.17 m/s (half a step
        # glided over each 0.86 s beat); was 0.3 for the old ~1 m/s pattern.
        velocity_tracking_width = 0.15
        # Provisional: width of the exp(-(omega_z / width)^2) yaw-rate reward
        # [rad/s]. At 0.3, the 0.28 rad/s of run Nov13_02-15-31_ scores 0.42.
        yaw_rate_tracking_width = 0.3
        # Provisional: width of the exp(-|Δx - (d/2)·p·e|^2 / width^2) per-beat
        # displacement reward [m]. The body moves 9 cm per moving beat; at
        # 0.05 m, standing still scores 0.04 (was 0.10 for the old pattern).
        displacement_tracking_width = 0.05
        # Provisional: width of the exp(-err^2 / width^2) per-foot footfall
        # reward [m] (world-frame foot travel since the beat began).
        footfall_tracking_width = 0.05
        # Provisional: width of the exp(-err^2 / width^2) stance-foot placement
        # reward [m]. Run Nov13_15-31-54_ had a 16 cm median error (scores 0.26
        # mean); the reference alone has 9 cm from joint lag (scores 0.42).
        foot_placement_width = 0.10

    class asset(Go2TrotCfg.asset):
        # Head_upper and Head_lower are separate bodies from base on the Go2.
        # Run Nov10_21-15-58_ learned to rest its face on the floor at 40 deg
        # pitch because only base contact ended an episode.
        terminate_after_contacts_on = ["base", "Head"]

    class push_robots(Go2TrotCfg.push_robots):
        toggle = False  # no pushes until the box works

    class reward_settings(Go2TrotCfg.reward_settings):
        # Provisional starting value: vertical foot force that counts as contact
        # for touchdown counting, support reward and slip penalty [N]
        # (about 3 % of body weight).
        foot_contact_force_n = 5.0
        # Base height the min_base_height reward saturates at [m]: the reset
        # height. The reference-only run sagged to about 0.20 m with nothing
        # rewarding the body for staying up.
        base_height_target = STANDING_HEIGHT_M + FOOT_RADIUS_M

    class scaling(Go2TrotCfg.scaling):
        # Same scale as base_lin_vel so the critic sees both in the same units.
        reference_body_velocity = BASE_HEIGHT_REF
        beat_displacement = 0.3  # about one step [m]
        beat_foot_displacement = 0.3  # about one step [m]


class Go2BoxStepRunnerCfg(Go2TrotRunnerCfg):
    class actor(Go2TrotRunnerCfg.actor):
        # 52 numbers. No velocity or height commands, no foot contacts.
        obs = [
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "step_oscillator_obs",
            "box_phase_obs",
            "step_size",
        ]

    class critic(Go2TrotRunnerCfg.critic):
        obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "dof_pos_obs",
            "dof_vel",
            "dof_pos_target",
            "phase_obs",
            "phase_frequency",
            "step_oscillator_obs",
            "box_phase_obs",
            "step_size",
            "reference_body_velocity",
            "beat_displacement",
            "beat_foot_displacement",
        ]

        class reward(Go2TrotRunnerCfg.critic.reward):
            # Provisional starting values, modeled on Go2TrotRunnerCfg. None of
            # these weights has been decided by the user; review before training.
            class weights(Go2TrotRunnerCfg.critic.reward.weights):
                # Off: no velocity or height commands in this task.
                tracking_lin_vel = 0.0
                tracking_base_height = 0.0
                trot_support = 0.0
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
                # of each step). Lowered from 4.0 on 2026-11-14 so leaning toward
                # the corner pays less than stepping there. Provisional.
                tracking_beat_displacement = 2.0
                # Feet land where scheduled and planted feet stay put (world
                # frame). Run Nov14_20-08-26_ shifted its body without real steps.
                tracking_footfalls = 4.0
                # Each planted foot off the ground costs this much per step. Run
                # Nov14_20-41-13_ bounded (all four feet off 44 % of the time).
                # Provisional.
                planted_foot_lift = 1.0
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

    class runner(Go2TrotRunnerCfg.runner):
        experiment_name = "go2boxstep"
