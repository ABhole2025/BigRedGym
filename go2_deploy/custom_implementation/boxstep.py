# AI-assisted: deploy hooks for the go2boxstep task, written from the course
# template go2_deploy/custom_implementation/go2trot_base.py. The beat clock,
# clock observations and joint reference come from the same boxstep_reference
# module the training task uses, so robot and simulator share one copy.
"""Go2 waltz box step: custom deploy implementation (group "boxstep").

Team: James Thomforde (jht224), Max Kinder (dmk299), Abha Bhole (ab2792).

Notes for course staff
----------------------
- Task: go2boxstep. Train for 1000 iterations
  (``uv run --frozen scripts/train.py --task=go2boxstep --headless``).
- Beat clock: starts when the controller enters CUSTOM_CTRL (press "c" / A on
  beat 1 of the music) and runs at TEMPO_BPM. Before that it holds beat 1.
- **Action space is customized.** The policy outputs corrections on top of a
  time-varying reference (absolute joint angles from boxstep_reference), and the
  training default posture is all zeros. The deploy code applies targets as
  ``action + cfg.default_dof_pos``, so update_custom_buffers writes the
  current reference into ``cfg.default_dof_pos`` every control step. For the
  same reason ``dof_pos_obs`` is overridden to the raw joint angles (training
  observes q - 0).
- **PD gains differ.** Training uses kp = 20, kd = 0.5 (randomized +/-10 % and
  +/-20 %); DeployConfig defaults to kp = 30, kd = 2. Please run at kp = 20,
  kd = 0.5 if possible.
- **Action smoothing.** Training has none; DeployConfig applies an EMA
  (0.095 s). It smooths only the corrections, not the reference.
- Observation scaling: Go2BoxStepCfg.scaling (adds tempo_bpm = 100).
"""

import torch

from go2_deploy.state import State
from gym.envs.go2 import boxstep_reference as reference
from gym.envs.go2.go2boxstep_config import Go2BoxStepCfg

SETTINGS = Go2BoxStepCfg.boxstep
TEMPO_BPM = 70.0  # must lie inside SETTINGS.tempo_bpm_range
STEP_SIZE_M = 0.25  # smallest trained step, inside SETTINGS.step_size_range

# Unitree motor order -> canonical FL, FR, RL, RR (same as deploy_utility).
UNITREE_TO_CANONICAL_JOINT_IDX = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])


class CustomDeployConfig:
    # Same order as Go2BoxStepRunnerCfg.actor.obs (52 numbers).
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

    # Length of observation vector pieces
    custom_obs_sizes = {
        "beat_clock_obs": 2,
        "tempo_bpm": 1,
        "beat_of_box_obs": reference.BEATS_PER_BOX,
        "step_size": 1,
    }

    class DeployScaling(Go2BoxStepCfg.scaling):
        pass


def compute_reference_angles(beat_time_s):
    """Return the absolute reference joint angles (12,) at beat_time_s."""
    return reference.compute_reference_joint_angles(
        torch.tensor([beat_time_s]),
        torch.tensor([60.0 / TEMPO_BPM]),
        SETTINGS.swing_fraction,
        torch.tensor([STEP_SIZE_M]),
        SETTINGS.standing_height_m,
        SETTINGS.swing_clearance_m,
    )[0]


# Initialize any custom buffers your implementation uses (for 1 environment)
def get_custom_buffers_dict():
    buffer_dict = {
        "beat_time": 0.0,  # seconds since beat 1 of the box
        "beat_clock_start": -1.0,  # monotonic time of beat 1; -1 = not started
        "tempo_bpm": TEMPO_BPM,
        "step_size": STEP_SIZE_M,
        "gait_reference": compute_reference_angles(0.0),
    }
    return buffer_dict


# Define how your custom buffers should be updated at the control frequency
def update_custom_buffers(main_controller, t):
    running = getattr(main_controller, "_state", None) == State.CUSTOM_CTRL
    if running and main_controller.beat_clock_start < 0.0:
        main_controller.beat_clock_start = t
    elif not running:
        main_controller.beat_clock_start = -1.0
    main_controller.beat_time = t - main_controller.beat_clock_start if running else 0.0
    main_controller.gait_reference = compute_reference_angles(main_controller.beat_time)
    # Targets are applied as action + default_dof_pos: make that the reference.
    main_controller.cfg.default_dof_pos[:] = main_controller.gait_reference


# Return a dict mapping observation names to the functions that compute them
def get_custom_obs_pieces():
    obs_pieces = {
        "dof_pos_obs": _get_obs_dof_pos_obs,
        "beat_clock_obs": _get_obs_beat_clock,
        "tempo_bpm": _get_obs_tempo_bpm,
        "beat_of_box_obs": _get_obs_beat_of_box,
        "step_size": _get_obs_step_size,
    }
    return obs_pieces


def _compute_beat_state(main_controller):
    return reference.compute_beat_state(
        torch.tensor([main_controller.beat_time]),
        torch.tensor([60.0 / main_controller.tempo_bpm]),
    )


# Raw joint angles: the training default posture is zero (see notes above).
def _get_obs_dof_pos_obs(main_controller, lowstate_msg):
    motor_states = lowstate_msg.motor_state
    joint_angles = torch.tensor([motor_states[i].q for i in range(12)])
    return joint_angles[UNITREE_TO_CANONICAL_JOINT_IDX]


def _get_obs_beat_clock(main_controller, lowstate_msg):
    _, beat_fraction = _compute_beat_state(main_controller)
    return reference.compute_beat_clock(beat_fraction)[0]


def _get_obs_tempo_bpm(main_controller, lowstate_msg):
    return torch.tensor([main_controller.tempo_bpm])


def _get_obs_beat_of_box(main_controller, lowstate_msg):
    beat_index, _ = _compute_beat_state(main_controller)
    return reference.compute_beat_of_box(beat_index)[0]


def _get_obs_step_size(main_controller, lowstate_msg):
    return torch.tensor([main_controller.step_size])
