import math
import torch

from gym.envs.go2.go2boxstep_config import Go2BoxStepCfg
from gym.envs.go2 import boxstep_reference as reference


class CustomDeployConfig:
    # Must match Go2BoxStepRunnerCfg.actor.obs exactly.
    obs_vector = [
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

    custom_obs_sizes = {
        "phase_obs": 2,
        "phase_frequency": 1,
        "step_oscillator_obs": 4,
        "box_phase_obs": 2,
        "step_size": 1,
    }

    class DeployScaling(Go2BoxStepCfg.scaling):
        pass


def get_custom_buffers_dict():
    bpm = 70.0

    return {
        "beat_time": 0.0,
        "beat_period": 60.0 / bpm,
        "step_size": 0.25,
        "phase": 0.0,
        "phase_frequency": bpm / 60.0,
    }


def update_custom_buffers(main_controller, t):
    # Assumes t is elapsed time since starting the controller.
    main_controller.beat_time = t

    beat_period = main_controller.beat_period
    beat_fraction = (t / beat_period) % 1.0

    main_controller.phase = 2.0 * math.pi * beat_fraction


def get_custom_obs_pieces():
    return {
        "phase_obs": _get_obs_phase_obs,
        "phase_frequency": _get_obs_phase_frequency,
        "step_oscillator_obs": _get_obs_step_oscillator_obs,
        "box_phase_obs": _get_obs_box_phase_obs,
        "step_size": _get_obs_step_size,
    }


def _get_obs_phase_obs(main_controller, lowstate_msg):
    phase = main_controller.phase
    return torch.tensor([math.sin(phase), math.cos(phase)])


def _get_obs_phase_frequency(main_controller, lowstate_msg):
    return torch.tensor([main_controller.phase_frequency])


def _get_obs_step_oscillator_obs(main_controller, lowstate_msg):
    beat_index, beat_fraction = reference.compute_beat_state(
        torch.tensor([main_controller.beat_time]),
        torch.tensor([main_controller.beat_period]),
    )

    swing_fraction = torch.tensor([
        Go2BoxStepCfg.boxstep.swing_time_s
        / main_controller.beat_period
    ])

    step_phase = reference.compute_step_oscillator(
        beat_index, beat_fraction, swing_fraction
    )

    pair_phases = reference.compute_pair_phases(
        beat_index, step_phase
    )

    return torch.stack(
        (torch.sin(pair_phases), torch.cos(pair_phases)),
        dim=2,
    ).reshape(4)


def _get_obs_box_phase_obs(main_controller, lowstate_msg):
    box_phase = reference.compute_box_phase(
        torch.tensor([main_controller.beat_time]),
        torch.tensor([main_controller.beat_period]),
    )

    return torch.stack(
        (torch.sin(box_phase), torch.cos(box_phase)),
        dim=1,
    ).reshape(2)


def _get_obs_step_size(main_controller, lowstate_msg):
    return torch.tensor([main_controller.step_size])