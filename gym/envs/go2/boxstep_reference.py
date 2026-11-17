# AI-assisted: box-step reference math (beat clock, step oscillator, foot paths,
# leg inverse kinematics) written from BOXSTEP_MATH.md Parts 1-3 and the function
# table in BOXSTEP_PROJECT_MAP.md, so training and deploy can share one copy.
# Schedule rewritten 2026-11-14: one leg pair steps per beat ("step, step, wait").
"""Waltz box-step reference for the Go2.

Pure torch, no simulator imports. Every function is batched over N robots.

The box is four bars of three beats (12 beats). In each bar the body moves one
step d along that bar's direction: beat 1 the lead pair steps d, beat 2 the
follow pair steps d, beat 3 all four feet wait. The body glides d/2 over each
of the two moving beats, rest to rest.

Symbols follow BOXSTEP_MATH.md:

    T_b  beat period (s)             k  beat within the box, 0..11
    s    beat fraction, 0 <= s < 1   sigma  swing fraction, T_sw / T_b
    u    swing progress of the pair stepping this beat, 0..1
    q    a pair's ground progress over the bar, 0..1
    P    body progress over the bar, 0..1
    d    step size (m)               h  standing height, g_c swing clearance

Leg order is FL, FR, RL, RR; joint order is hip, thigh, calf. Pair order is
left, right. Body frame: x forward, y left, z up.
"""

import math

import torch

BEATS_PER_BAR = 3
BARS_PER_BOX = 4
BEATS_PER_BOX = BEATS_PER_BAR * BARS_PER_BOX

# Bar schedule. Row b is bar b of the box.
BAR_DIRECTIONS = torch.tensor(
    [
        [1.0, 0.0],  # forward
        [0.0, -1.0],  # right
        [-1.0, 0.0],  # back
        [0.0, 1.0],  # left
    ]
)
# Lead side per bar: forward and left lead with the left pair, right and back
# with the right pair (the side of travel leads on sideways bars).
BAR_LEADS_WITH_LEFT = torch.tensor([True, False, False, True])

# Per-beat tables (12 rows), derived from the bar schedule.
_BEAT_IN_BAR = torch.arange(BEATS_PER_BOX) % BEATS_PER_BAR
_BAR_OF_BEAT = torch.arange(BEATS_PER_BOX) // BEATS_PER_BAR
BEAT_IS_MOVING = _BEAT_IN_BAR < 2
# Body motion direction on each beat; zero on the wait beat.
BEAT_DIRECTIONS = BAR_DIRECTIONS[_BAR_OF_BEAT] * BEAT_IS_MOVING.unsqueeze(1)
# Pair that steps on each beat: 0 = left, 1 = right, -1 = none (wait beat).
_LEAD_PAIR = torch.where(BAR_LEADS_WITH_LEFT, 0, 1)[_BAR_OF_BEAT]
BEAT_STEPPING_PAIR = torch.where(
    _BEAT_IN_BAR == 0, _LEAD_PAIR, torch.where(_BEAT_IN_BAR == 1, 1 - _LEAD_PAIR, -1)
)

# Leg geometry from go2.urdf, MATH 3.1.
HIP_OFFSET_M = 0.0955
LINK_LENGTH_M = 0.213
CALF_STRAIGHTEST_RAD = -0.83776
CALF_MOST_BENT_RAD = -2.7227
LEG_SIDE_SIGN = torch.tensor([1.0, -1.0, 1.0, -1.0])
# Which pair (0 = left, 1 = right) each leg belongs to: the pace gait.
LEG_PAIR_INDEX = torch.tensor([0, 1, 0, 1])


def _lookup(table, beat_index):
    return table.to(beat_index.device)[beat_index]


def _smooth_step(progress):
    """Rest-to-rest profile (1 - cos(pi x)) / 2 for x in [0, 1]."""
    return 0.5 * (1.0 - torch.cos(math.pi * progress))


def compute_beat_state(time_s, beat_period_s):
    """Return (beat_index (N,) long in 0..11, beat_fraction (N,) in [0, 1))."""
    beats_elapsed = time_s / beat_period_s
    beats_completed = torch.floor(beats_elapsed)
    beat_fraction = beats_elapsed - beats_completed
    beat_index = torch.remainder(beats_completed, BEATS_PER_BOX).long()
    return beat_index, beat_fraction


def compute_stepping_feet(beat_index):
    """Return (N, 4) bool: True for the feet of the pair stepping this beat."""
    stepping_pair = _lookup(BEAT_STEPPING_PAIR, beat_index)
    leg_pair = LEG_PAIR_INDEX.to(beat_index.device)
    return leg_pair.unsqueeze(0) == stepping_pair.unsqueeze(1)


def compute_step_oscillator(beat_index, beat_fraction, swing_fraction):
    """Return theta (N,) in [0, 2*pi) for the pair stepping this beat.

    theta runs 0 -> pi during the swing (0 <= s < sigma) and pi -> 2*pi over
    the rest of the beat, so sin(theta) > 0 exactly while the pair swings.
    Zero on the wait beat.
    """
    swing_part = math.pi * torch.clamp(beat_fraction / swing_fraction, max=1.0)
    rest_part = math.pi * torch.clamp(
        (beat_fraction - swing_fraction) / (1.0 - swing_fraction), 0.0, 1.0
    )
    step_phase = (swing_part + rest_part) * _lookup(BEAT_IS_MOVING, beat_index)
    return torch.remainder(step_phase, 2.0 * math.pi)


def compute_pair_phases(beat_index, step_phase):
    """Return each pair's oscillator phase (N, 2), left then right.

    The pair stepping this beat carries theta; the other pair is 0.
    """
    stepping_pair = _lookup(BEAT_STEPPING_PAIR, beat_index)
    left_phase = torch.where(stepping_pair == 0, step_phase, 0.0)
    right_phase = torch.where(stepping_pair == 1, step_phase, 0.0)
    return torch.stack((left_phase, right_phase), dim=1)


def compute_box_phase(time_s, beat_period_s):
    """Return the position in the 12-beat box (N,) as an angle in [0, 2*pi)."""
    boxes_elapsed = time_s / (BEATS_PER_BOX * beat_period_s)
    return 2.0 * math.pi * torch.remainder(boxes_elapsed, 1.0)


def compute_swing_progress(beat_index, beat_fraction, swing_fraction):
    """Return u (N, 2) in [0, 1] for the left and right pair.

    Only the pair stepping this beat is non-zero; it is clamp(s / sigma).
    """
    progress = torch.clamp(beat_fraction / swing_fraction, 0.0, 1.0)
    stepping_pair = _lookup(BEAT_STEPPING_PAIR, beat_index)
    left = torch.where(stepping_pair == 0, progress, 0.0)
    right = torch.where(stepping_pair == 1, progress, 0.0)
    return torch.stack((left, right), dim=1)


def compute_pair_ground_progress(beat_index, beat_fraction, swing_fraction):
    """Return q (N, 2): each pair's ground progress over the bar, 0 -> 1.

    A pair is at 0 before its beat, follows the smooth swing profile during
    its swing, and stays at 1 for the rest of the bar.
    """
    beat_in_bar = beat_index % BEATS_PER_BAR
    swing = _smooth_step(torch.clamp(beat_fraction / swing_fraction, 0.0, 1.0))
    lead_progress = torch.where(beat_in_bar == 0, swing, 1.0)
    follow_progress = torch.where(
        beat_in_bar == 0, 0.0, torch.where(beat_in_bar == 1, swing, 1.0)
    )
    leads_with_left = _lookup(BAR_LEADS_WITH_LEFT, beat_index // BEATS_PER_BAR)
    left = torch.where(leads_with_left, lead_progress, follow_progress)
    right = torch.where(leads_with_left, follow_progress, lead_progress)
    return torch.stack((left, right), dim=1)


def compute_body_progress(beat_index, beat_fraction):
    """Return P (N,): body progress over the bar, 0 -> 1.

    The body glides half a step over each moving beat, rest to rest, and
    stays put on the wait beat.
    """
    beat_in_bar = (beat_index % BEATS_PER_BAR).to(beat_fraction.dtype)
    within_beat = _smooth_step(beat_fraction)
    return torch.where(beat_in_bar < 2, 0.5 * (beat_in_bar + within_beat), 1.0)


def compute_foot_offsets(
    beat_index, beat_fraction, swing_fraction, step_size_m, swing_clearance_m
):
    """Return each foot's offset from neutral (N, 4, 3) in metres.

    Horizontal: r = d * (q - P) * e_bar. Vertical: g_c * sin(pi * u).
    """
    leg_pair = LEG_PAIR_INDEX.to(beat_index.device)
    ground_progress = compute_pair_ground_progress(
        beat_index, beat_fraction, swing_fraction
    )[:, leg_pair]
    body_progress = compute_body_progress(beat_index, beat_fraction)
    direction = _lookup(BAR_DIRECTIONS, beat_index // BEATS_PER_BAR)

    relative_progress = ground_progress - body_progress.unsqueeze(1)
    horizontal_offset = (
        step_size_m.reshape(-1, 1, 1)
        * relative_progress.unsqueeze(2)
        * direction.unsqueeze(1)
    )
    swing_progress = compute_swing_progress(beat_index, beat_fraction, swing_fraction)
    lift = swing_clearance_m * torch.sin(math.pi * swing_progress[:, leg_pair])
    return torch.cat((horizontal_offset, lift.unsqueeze(2)), dim=2)


def compute_scheduled_stance(beat_index, beat_fraction, swing_fraction):
    """Return (N, 4) bool, True where a foot is scheduled to be on the ground."""
    swinging_now = (beat_fraction < swing_fraction).unsqueeze(1)
    return ~(compute_stepping_feet(beat_index) & swinging_now)


def compute_beat_body_shift(beat_index, beat_fraction, step_size_m):
    """Return the body's target displacement since this beat began (N, 2), m.

    (d / 2) * p(s) * e_k on moving beats; zero on the wait beat.
    """
    direction = _lookup(BEAT_DIRECTIONS, beat_index)
    shift = 0.5 * step_size_m * _smooth_step(beat_fraction)
    return shift.unsqueeze(1) * direction


def compute_body_velocity(
    beat_index, beat_fraction, swing_fraction, step_size_m, beat_period_s
):
    """Return the reference body velocity (N, 2) in m/s: d/dt of the body shift.

    swing_fraction is unused (kept so callers did not change).
    """
    progress_rate_per_beat = 0.5 * math.pi * torch.sin(math.pi * beat_fraction)
    speed = 0.5 * step_size_m * progress_rate_per_beat / beat_period_s
    return speed.unsqueeze(1) * _lookup(BEAT_DIRECTIONS, beat_index)


def compute_neutral_foot_positions(standing_height_m, device="cpu"):
    """Return each foot's neutral position (4, 3) in its own hip frame.

    Directly below the thigh joint at the standing height: (0, side * l1, -h).
    """
    side_sign = LEG_SIDE_SIGN.to(device)
    return torch.stack(
        (
            torch.zeros(4, device=device),
            side_sign * HIP_OFFSET_M,
            torch.full((4,), -standing_height_m, device=device),
        ),
        dim=1,
    )


def solve_leg_joint_angles(foot_position_m):
    """Return joint angles (N, 12) for foot targets (N, 4, 3) in each hip frame.

    MATH 3.3-3.5. The knee is clamped to its joint limits, which also caps the
    leg at the reach limit; the leg then points at the target but falls short.
    """
    foot_x, foot_y, foot_z = foot_position_m.unbind(dim=2)
    side_offset = LEG_SIDE_SIGN.to(foot_position_m.device) * HIP_OFFSET_M

    leg_plane_length = torch.sqrt(
        torch.clamp(foot_y**2 + foot_z**2 - HIP_OFFSET_M**2, min=0.0)
    )
    hip_angle = torch.atan2(
        leg_plane_length * foot_y + side_offset * foot_z,
        side_offset * foot_y - leg_plane_length * foot_z,
    )

    thigh_to_foot_squared = foot_x**2 + leg_plane_length**2
    knee_cosine = torch.clamp(
        thigh_to_foot_squared / (2.0 * LINK_LENGTH_M**2) - 1.0,
        min=math.cos(CALF_MOST_BENT_RAD),
        max=math.cos(CALF_STRAIGHTEST_RAD),
    )
    calf_angle = -torch.arccos(knee_cosine)
    thigh_angle = torch.atan2(-foot_x, leg_plane_length) - 0.5 * calf_angle

    joint_angles = torch.stack((hip_angle, thigh_angle, calf_angle), dim=2)
    return joint_angles.reshape(foot_position_m.shape[0], 12)


def compute_foot_positions(joint_angles):
    """Return each foot's position (N, 4, 3) in its hip frame from (N, 12) angles.

    Forward kinematics, MATH 3.2. The inverse of solve_leg_joint_angles.
    """
    hip, thigh, calf = joint_angles.reshape(-1, 4, 3).unbind(dim=2)
    side_offset = LEG_SIDE_SIGN.to(joint_angles.device) * HIP_OFFSET_M
    foot_x = -LINK_LENGTH_M * (torch.sin(thigh) + torch.sin(thigh + calf))
    plane_z = -LINK_LENGTH_M * (torch.cos(thigh) + torch.cos(thigh + calf))
    foot_y = side_offset * torch.cos(hip) - plane_z * torch.sin(hip)
    foot_z = side_offset * torch.sin(hip) + plane_z * torch.cos(hip)
    return torch.stack((foot_x, foot_y, foot_z), dim=2)


def compute_reference_foot_positions(
    time_s,
    beat_period_s,
    swing_time_s,
    step_size_m,
    standing_height_m,
    swing_clearance_m,
):
    """Return the reference foot targets (N, 4, 3), each in its own hip frame."""
    beat_index, beat_fraction = compute_beat_state(time_s, beat_period_s)
    swing_fraction = swing_time_s / beat_period_s
    foot_offsets = compute_foot_offsets(
        beat_index, beat_fraction, swing_fraction, step_size_m, swing_clearance_m
    )
    neutral_feet = compute_neutral_foot_positions(standing_height_m, time_s.device)
    return neutral_feet.unsqueeze(0) + foot_offsets


def compute_reference_joint_angles(
    time_s,
    beat_period_s,
    swing_time_s,
    step_size_m,
    standing_height_m,
    swing_clearance_m,
):
    """Return the absolute reference joint angles (N, 12) for the whole robot.

    time_s, beat_period_s and step_size_m are (N,) tensors; swing_time_s,
    standing_height_m and swing_clearance_m are floats.
    """
    foot_targets = compute_reference_foot_positions(
        time_s,
        beat_period_s,
        swing_time_s,
        step_size_m,
        standing_height_m,
        swing_clearance_m,
    )
    return solve_leg_joint_angles(foot_targets)
