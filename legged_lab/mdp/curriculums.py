# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from legged_lab.envs.base.command_curriculum import GridAdaptiveCurriculum


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


# -----------------------------------------------------------------------------
# Grid-adaptive command curriculum (Isaac Lab friendly)
# -----------------------------------------------------------------------------

def grid_adaptive_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_names: Sequence[str],
    success_thresholds: Sequence[float],
) -> None:
    """Grid-based adaptive command curriculum that runs on reset.

    - Keeps per-env bin indices and a shared GridAdaptiveCurriculum instance on the env.
    - Uses configurable reward terms to decide success; falls back to zero if terms missing.
    - Samples new commands for the resetting envs and logs means for TensorBoard.
    """

    if len(env_ids) == 0:
        return

    # Lazy init shared state
    if not hasattr(env, "_grid_cmd_curriculum"):
        cfg = getattr(env.cfg.commands, "command_curriculum_cfg", None)
        if cfg is None:
            return
        env._grid_cmd_curriculum = GridAdaptiveCurriculum(cfg)
        env._grid_cmd_bins = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    curriculum = env._grid_cmd_curriculum

    # Collect per-term episode sums for the resetting envs
    task_rewards = []
    for name in reward_term_names:
        term_sums = env.reward_manager._episode_sums.get(name, None)
        if term_sums is None:
            task_rewards.append(torch.zeros(len(env_ids), device=env.device))
        else:
            task_rewards.append(term_sums[env_ids].detach().clone())

    # Align thresholds length
    thresholds = list(success_thresholds)
    if len(thresholds) < len(task_rewards):
        thresholds += [thresholds[-1]] * (len(task_rewards) - len(thresholds))

    if len(task_rewards) > 0:
        curriculum.update_success_rate(
            env._grid_cmd_bins[env_ids].cpu().numpy(),
            task_rewards,
            thresholds[: len(task_rewards)],
        )
        curriculum.update_weights()

    # Sample new commands for the resetting envs
    cmds_np, bin_inds = curriculum.sample(len(env_ids))
    cmds = torch.tensor(cmds_np, device=env.device, dtype=torch.float)
    env.command_generator.command[env_ids, 0] = cmds[:, 0]
    env.command_generator.command[env_ids, 1] = cmds[:, 1]
    env.command_generator.command[env_ids, 2] = cmds[:, 2]
    env._grid_cmd_bins[env_ids] = torch.as_tensor(bin_inds, device=env.device, dtype=torch.long)

    # TensorBoard-friendly logs
    if not hasattr(env, "extras"):
        env.extras = {}
    env.extras.setdefault("log", {})
    env.extras["log"]["Curriculum/max_cmd_x"] = env.command_generator.command[:, 0].max()
    env.extras["log"]["Curriculum/min_cmd_x"] = env.command_generator.command[:, 0].min()
    env.extras["log"]["Curriculum/max_cmd_y"] = env.command_generator.command[:, 1].max()
    env.extras["log"]["Curriculum/min_cmd_y"] = env.command_generator.command[:, 1].min()
    env.extras["log"]["Curriculum/max_cmd_yaw"] = env.command_generator.command[:, 2].max()
    env.extras["log"]["Curriculum/min_cmd_yaw"] = env.command_generator.command[:, 2].min()
    env.extras["log"]["Curriculum/weight_mean"] = torch.as_tensor(curriculum.weights.mean())
