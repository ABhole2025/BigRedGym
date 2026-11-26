# AI-assisted: Go2BoxStep task (buffers, beat clock, reference, observations,
# rewards) written from BOXSTEP_PROJECT_MAP.md. Built on Go2 since 2026-11-20
# (PR review): it only borrows the residual-on-a-reference control from Go2Trot.
import torch

from gym.envs.go2 import boxstep_reference as reference
from gym.envs.go2.go2 import Go2
from gym.utils.sampling import masked_update, torch_rand_float


class Go2BoxStep(Go2):
    """Waltz box step in a pace gait, driven by a beat clock.

    The reference supplies absolute joint angles in gait_reference (default joint
    angles are zero), and the policy adds dof_pos_target corrections on top. All
    beat-clock math lives in boxstep_reference, shared with the deploy code.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self._create_boxstep_buffers()
        self._update_clock_observations()
        self._update_gait_reference()

    def _create_boxstep_buffers(self):
        settings = self.cfg.boxstep

        def create_column(width=1, fill=0.0, dtype=torch.float):
            return torch.full(
                (self.num_envs, width), fill, dtype=dtype, device=self.device
            )

        nominal_tempo_bpm = sum(settings.tempo_bpm_range) / 2.0
        nominal_step_size_m = sum(settings.step_size_range) / 2.0
        # Beat clock state: time since beat 1 of the box, and the beat period.
        self.beat_time = create_column()
        self.beat_period = create_column(fill=60.0 / nominal_tempo_bpm)
        self.step_size = create_column(fill=nominal_step_size_m)
        # Clock observations: music clock (sin, cos), tempo, beat of the box.
        self.beat_clock_obs = create_column(2)
        self.tempo_bpm = create_column(fill=nominal_tempo_bpm)
        self.beat_of_box_obs = create_column(reference.BEATS_PER_BOX)
        # Reference outputs.
        self.gait_reference = torch.zeros_like(self.dof_pos_target)
        self.reference_foot_positions = torch.zeros(
            self.num_envs, 4, 3, dtype=torch.float, device=self.device
        )
        self.scheduled_stance = create_column(4, True, torch.bool)
        self.reference_body_velocity = create_column(2)
        # Touchdown counting.
        self.touchdown_count = create_column(4)
        self.previous_foot_contact = create_column(4, True, torch.bool)
        self.new_touchdown = create_column(4, False, torch.bool)
        # Body position and heading when the current beat started (world xy),
        # and the displacement since then in that start frame (critic only).
        self.beat_start_position = create_column(2)
        self.beat_start_yaw = create_column()
        self.beat_displacement = create_column(2)

        self._actuated_dof_pos_limits = self.dof_pos_limits.index_select(
            0, self.actuated_dof_indices
        )
        neutral_feet = reference.compute_neutral_foot_positions(
            settings.standing_height_m, self.device
        )
        standing_angles = reference.solve_leg_joint_angles(neutral_feet.unsqueeze(0))
        self._standing_dof_pos = torch.zeros(1, self.num_dof, device=self.device)
        self._standing_dof_pos[:, self.actuated_dof_indices] = standing_angles

    # ------------------------------------------------------
    #
    # Clock and reference
    #
    # ------------------------------------------------------

    def _compute_beat_state(self):
        """Return (beat_index, beat_fraction) (N,) from the shared beat clock."""
        return reference.compute_beat_state(
            self.beat_time.squeeze(1), self.beat_period.squeeze(1)
        )

    def _post_physics_step(self):
        super()._post_physics_step()
        # Runs once per physics substep.
        self.beat_time.add_(self.dt / self.cfg.control.decimation)

    def _update_clock_observations(self):
        beat_index, beat_fraction = self._compute_beat_state()
        self.beat_clock_obs[:] = reference.compute_beat_clock(beat_fraction)
        self.beat_of_box_obs[:] = reference.compute_beat_of_box(beat_index)

    def _update_gait_reference(self):
        settings = self.cfg.boxstep
        beat_index, beat_fraction = self._compute_beat_state()
        step_size = self.step_size.squeeze(1)
        self.reference_foot_positions[:] = reference.compute_reference_foot_positions(
            self.beat_time.squeeze(1),
            self.beat_period.squeeze(1),
            settings.swing_fraction,
            step_size,
            settings.standing_height_m,
            settings.swing_clearance_m,
        )
        self.gait_reference[:] = reference.solve_leg_joint_angles(
            self.reference_foot_positions
        )
        self.scheduled_stance[:] = reference.compute_scheduled_stance(
            beat_index, beat_fraction, settings.swing_fraction
        )
        self.reference_body_velocity[:] = reference.compute_body_velocity(
            beat_index,
            beat_fraction,
            settings.swing_fraction,
            step_size,
            self.beat_period.squeeze(1),
        )

    def _pre_decimation_step(self):
        self._update_gait_reference()
        reference_pos = self.gait_reference + self.default_dof_pos.index_select(
            1, self.actuated_dof_indices
        )
        # Bound the full PD target (reference + correction) to the joint limits;
        # keep the applied correction so observations and action rewards see it.
        self.dof_pos_target.clamp_(
            min=self._actuated_dof_pos_limits[:, 0] - reference_pos,
            max=self._actuated_dof_pos_limits[:, 1] - reference_pos,
        )

    def _compute_torques(self):
        """PD on reference + policy correction (+ default, zero for this task)."""
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        default_pos = self.default_dof_pos.index_select(1, self.actuated_dof_indices)
        torques = (
            self.p_gains
            * (self.gait_reference + self.dof_pos_target + default_pos - pos)
            + self.d_gains * (self.dof_vel_target - vel)
            + self.tau_ff
        )
        return torch.clip(
            torques, -self.actuated_torque_limits, self.actuated_torque_limits
        ).view(self.torques.shape)

    # ------------------------------------------------------
    #
    # Reset and commands
    #
    # ------------------------------------------------------

    def reset_to_basic(self, reset_mask):
        """Reset to the standing pose (leg IK of the neutral feet)."""
        super().reset_to_basic(reset_mask)
        masked_update(self.dof_pos, self._standing_dof_pos, reset_mask)

    def _reset_system(self, reset_mask):
        super()._reset_system(reset_mask)
        settings = self.cfg.boxstep
        tempo_bpm = torch_rand_float(
            *settings.tempo_bpm_range, (self.num_envs, 1), device=self.device
        )
        step_size = torch_rand_float(
            *settings.step_size_range, (self.num_envs, 1), device=self.device
        )
        starting_beat = torch.randint(
            reference.BEATS_PER_BOX, (self.num_envs, 1), device=self.device
        )
        beat_period = 60.0 / tempo_bpm
        masked_update(self.beat_period, beat_period, reset_mask)
        masked_update(self.tempo_bpm, tempo_bpm, reset_mask)
        masked_update(self.step_size, step_size, reset_mask)
        # Episodes start on a beat boundary of a random beat of the box.
        masked_update(self.beat_time, starting_beat * beat_period, reset_mask)
        self.touchdown_count.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self.previous_foot_contact.masked_fill_(reset_mask.unsqueeze(1), True)
        self.new_touchdown.masked_fill_(reset_mask.unsqueeze(1), False)
        self._record_beat_start(reset_mask)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        # Corrections start at zero, on top of the reference.
        self.dof_pos_target.masked_fill_(reset_mask.unsqueeze(1), 0.0)
        self._update_gait_reference()
        masked_update(
            self.dof_pos_history,
            self.dof_pos_target.tile(1, 3),
            reset_mask,
        )
        self._update_clock_observations()

    def _resample_commands(self, command_mask):
        """No velocity commands in the box step: commands stay zero."""
        return None

    # ------------------------------------------------------
    #
    # Contacts
    #
    # ------------------------------------------------------

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._update_clock_observations()
        # Refresh the schedule at the new time so rewards and contacts agree.
        self._update_gait_reference()
        self._count_touchdowns()
        self._record_beat_start(self._starts_new_beat())
        self._update_beat_displacement()

    def _starts_new_beat(self):
        """Return (N,) bool: True on the first control step of a beat."""
        _, beat_fraction = self._compute_beat_state()
        return beat_fraction * self.beat_period.squeeze(1) < self.dt

    def _compute_base_yaw(self):
        """Return the body's heading (N,) in rad from base_quat (x, y, z, w)."""
        x, y, z, w = self.base_quat.unbind(dim=1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _record_beat_start(self, mask):
        masked_update(self.beat_start_position, self.root_states[:, :2], mask)
        masked_update(self.beat_start_yaw, self._compute_base_yaw().unsqueeze(1), mask)
        self.beat_displacement.masked_fill_(mask.unsqueeze(1), 0.0)

    def _update_beat_displacement(self):
        """Body displacement since the beat started, in the beat-start frame."""
        world_shift = self.root_states[:, :2] - self.beat_start_position
        cos_yaw = torch.cos(self.beat_start_yaw.squeeze(1))
        sin_yaw = torch.sin(self.beat_start_yaw.squeeze(1))
        self.beat_displacement[:, 0] = (
            cos_yaw * world_shift[:, 0] + sin_yaw * world_shift[:, 1]
        )
        self.beat_displacement[:, 1] = (
            -sin_yaw * world_shift[:, 0] + cos_yaw * world_shift[:, 1]
        )

    def _detect_foot_contact(self):
        """Return (N, 4) bool: vertical foot force above the contact threshold."""
        vertical_force = self.contact_forces[:, self.feet_indices, 2]
        return vertical_force > self.cfg.reward_settings.foot_contact_force_n

    def _count_touchdowns(self):
        foot_contact = self._detect_foot_contact()
        self.new_touchdown[:] = foot_contact & ~self.previous_foot_contact
        # The first control step of each beat clears that beat's count.
        self.touchdown_count.masked_fill_(self._starts_new_beat().unsqueeze(1), 0.0)
        self.touchdown_count.add_(self.new_touchdown.float())
        self.previous_foot_contact[:] = foot_contact

    def _count_allowed_touchdowns(self):
        """Return (N, 1): one scheduled touchdown per foot on moving beats."""
        beat_index, _ = self._compute_beat_state()
        is_moving = reference.BEAT_IS_MOVING.to(self.device)[beat_index]
        return is_moving.float().unsqueeze(1)

    # ------------------------------------------------------
    #
    # Rewards
    #
    # ------------------------------------------------------

    def _foot_contact_strength(self):
        """Map upward foot load smoothly from zero to nominal body weight."""
        load = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2]
            / (9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)),
            min=0.0,
            max=1.0,
        )
        return load.square() * (3.0 - 2.0 * load)

    def _reward_scheduled_support(self):
        """1 when every foot scheduled as stance is in contact, else 0.

        All-or-nothing: run Nov13_01-51-32_ held one leg up because the earlier
        fraction-of-feet version still paid 0.75 on three feet.
        """
        missing = self.scheduled_stance & ~self._detect_foot_contact()
        return (~missing.any(dim=1)).float()

    def _reward_swing_contact(self):
        """Penalize load carried by feet scheduled as swing."""
        scheduled_swing = (~self.scheduled_stance).float()
        return -torch.mean(scheduled_swing * self._foot_contact_strength(), dim=1)

    def _reward_tracking_body_velocity(self):
        """Body velocity follows reference_body_velocity (body frame, xy)."""
        error = self.reference_body_velocity - self.base_lin_vel[:, :2]
        width = self.cfg.boxstep.velocity_tracking_width
        return torch.exp(-torch.sum(torch.square(error), dim=1) / width**2)

    def _reward_tracking_beat_displacement(self):
        """Body has moved d * p(s) along this beat's direction since the beat began.

        Position, not velocity: run Nov13_02-32-57_ covered 19-67 % of each step
        because the velocity reward is zero-target most of the time. By the end
        of a moving beat p = 1, so the target is the full step; on hold beats
        the direction is zero, so the target is to stay put.
        """
        beat_index, beat_fraction = self._compute_beat_state()
        body_progress = reference.compute_body_progress(
            beat_fraction, self.cfg.boxstep.swing_fraction
        )
        direction = reference.BEAT_DIRECTIONS.to(self.device)[beat_index]
        target = (self.step_size.squeeze(1) * body_progress).unsqueeze(1) * direction
        error = torch.sum(torch.square(self.beat_displacement - target), dim=1)
        width = self.cfg.boxstep.displacement_tracking_width
        return torch.exp(-error / width**2)

    def _reward_stance_foot_placement(self):
        """Feet scheduled as stance sit where the reference puts them under the hip.

        Horizontal (x, y) error in each hip frame, from the measured joint
        angles; exp(-err^2 / width^2), averaged over the stance feet. Run
        Nov13_15-31-54_ left the back-right foot about 20 cm behind its hip and
        22 cm under the body on beats 2-3.
        """
        joint_angles = self.dof_pos.index_select(1, self.actuated_dof_indices)
        feet = reference.compute_foot_positions(joint_angles)
        error = feet[..., :2] - self.reference_foot_positions[..., :2]
        width = self.cfg.boxstep.foot_placement_width
        score = torch.exp(-torch.sum(torch.square(error), dim=2) / width**2)
        stance = self.scheduled_stance.float()
        return torch.sum(score * stance, dim=1) / stance.sum(dim=1).clamp(min=1.0)

    def _reward_tracking_ang_vel(self):
        """Yaw rate held at zero: exp(-(omega_z / width)^2).

        Overrides Go2's version, which squares the error twice, so turning
        at 0.28 rad/s cost 0.06 % of the reward; run Nov13_02-15-31_ drifted
        5-44 deg clockwise over two boxes.
        """
        width = self.cfg.boxstep.yaw_rate_tracking_width
        return torch.exp(-torch.square(self.base_ang_vel[:, 2] / width))

    def _reward_stance_foot_slip(self):
        """Penalize horizontal speed of planted feet (scheduled stance, in contact)."""
        planted = self.scheduled_stance & self._detect_foot_contact()
        foot_speed_squared = torch.sum(
            torch.square(self._rigid_body_lin_vel[:, self.feet_indices, :2]), dim=2
        )
        return -torch.sum(planted * foot_speed_squared, dim=1)

    def _reward_extra_touchdowns(self):
        """Penalize each touchdown beyond the ones scheduled in this beat."""
        extra = self.new_touchdown & (
            self.touchdown_count > self._count_allowed_touchdowns()
        )
        return -extra.sum(dim=1).float()

    def _reward_correction_size(self):
        """Penalize large policy corrections, in units of the action scale."""
        normalized = self.dof_pos_target / self.scales["dof_pos_target"]
        return -torch.mean(torch.square(normalized), dim=1)
