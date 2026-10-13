"""Measure height tracking of the existing trained Go2Trot policy."""

import csv
import sys

import torch

from scripts.play import get_play_args, setup


def main():
    args = get_play_args([
        "--task", "go2trot",
        "--backend", "mujoco",
        "--device", "cpu",
        "--num_envs", "1",
        "--experiment_name", "a4_height_final",
        "--load_run", "Oct13_15-55-34_",
        "--checkpoint", "300",
        "--headless",
        "--no-keyboard",
        "--original_cfg",
    ])

    with torch.no_grad():
        env, runner = setup(args)

        # Keep locomotion commands at zero throughout this test.
        env.commands.zero_()

        steps_per_segment = 300
        targets = [0.18, 0.38, 0.18, 0.38]

        with open("height_tracking_test_most_recent.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s",
                "commanded_height_m",
                "actual_height_m",
            ])

            for target in targets:
                env.height_command.fill_(target)

                for _ in range(steps_per_segment):
                    runner.set_actions(
                        runner.actor_cfg["actions"],
                        runner.get_inference_actions(),
                        runner.actor_cfg["disable_actions"],
                    )
                    env.step()

                    writer.writerow([
                        env.common_step_counter * env.dt,
                        target,
                        env.base_height[0, 0].item(),
                    ])

                print(
                    f"target={target:.2f} m, "
                    f"final actual={env.base_height[0, 0].item():.3f} m"
                )

    print("Saved height_tracking_test_most_recent.csv")


if __name__ == "__main__":
    main()