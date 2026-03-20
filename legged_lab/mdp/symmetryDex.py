# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
import torch


# https://github.com/leggedrobotics/rsl_rl/issues/64 g1
# joint order ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
# 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
#  'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 
#  'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
#  'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint']

# self.robot.joint_names: ['hip_pitch_l_joint', 'hip_pitch_r_joint', 'waist_yaw_joint', 'hip_roll_l_joint', 'hip_roll_r_joint', 
#  'waist_roll_joint', 'hip_yaw_l_joint', 'hip_yaw_r_joint', 'waist_pitch_joint', 'knee_pitch_l_joint', 'knee_pitch_r_joint', 'shoulder_pitch_l_joint', 'shoulder_pitch_r_joint', 
#  'ankle_pitch_l_joint', 'ankle_pitch_r_joint', 'shoulder_roll_l_joint', 'shoulder_roll_r_joint', 'ankle_roll_l_joint', 'ankle_roll_r_joint', 'shoulder_yaw_l_joint', 'shoulder_yaw_r_joint', 
#  'elbow_pitch_l_joint', 'elbow_pitch_r_joint']


# 预计算镜像索引 - 模块级缓存，避免重复计算
_MIRROR_INDICES_CACHE = {}
ACTION_NUM = 23

def _get_mirror_indices(offset=0):
    """获取或创建镜像索引的缓存版本"""
    if offset not in _MIRROR_INDICES_CACHE:
        # 创建置换索引数组：对于23个关节，哪个位置应该从哪个位置读取
        perm = list(range(ACTION_NUM))
        # 交换对
        swap_pairs = [(0, 1), 
                      (3, 4), 
                      (6, 7), 
                      (9, 10), 
                      (11, 12), 
                    (13, 14), 
                    (15, 16), 
                    (17, 18), 
                    (19, 20), 
                    (21, 22)]
        for left, right in swap_pairs:
            perm[left], perm[right] = perm[right], perm[left]
        
        # 需要取反的索引
        negate_mask = [False] * ACTION_NUM
        negate_indices = [2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20]
        for idx in negate_indices:
            negate_mask[idx] = True
        
        _MIRROR_INDICES_CACHE[offset] = {
            'perm': torch.tensor(perm, dtype=torch.long),
            'negate': torch.tensor(negate_mask, dtype=torch.bool)
        }
    
    return _MIRROR_INDICES_CACHE[offset]


def mirror_joint_tensor(original: torch.Tensor, mirrored: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """快速镜像关节张量 - 使用预计算索引"""
    indices = _get_mirror_indices(offset)
    perm = indices['perm'].to(original.device)
    negate = indices['negate'].to(original.device)
    
    # 一次性置换所有关节
    joint_slice = slice(offset, offset + ACTION_NUM)
    mirrored[..., joint_slice] = original[..., offset + perm]
    # 一次性取反需要的关节
    mirrored[..., offset:offset + ACTION_NUM][..., negate] *= -1
    
def mirror_observation_policy(obs):
    """
    obs: (..., 840)  历史 10 帧，每帧 84 维
    return: (..., 1680)  原观测 + 镜像观测 沿最后一维拼接
    """
    if obs is None:
        return obs
    
    *batch_shape, _ = obs.shape
    batch_size = obs.shape[0] if batch_shape else 1
    
    # 预分配输出张量，避免 vstack
    result = torch.empty(batch_size * 2, 840, device=obs.device, dtype=obs.dtype)
    result[:batch_size] = obs
    
    # reshape 用于向量化操作
    obs_2d = obs.view(batch_size, 10, 84)
    flipped_2d = obs_2d.clone()
    
    # 向量化镜像所有帧（不用循环）
    # base ang vel x,z
    flipped_2d[..., 0] = -obs_2d[..., 0]
    flipped_2d[..., 2] = -obs_2d[..., 2]
    # projected gravity y
    flipped_2d[..., 4] = -obs_2d[..., 4]
    # velocity commands y/z
    flipped_2d[..., 7] = -obs_2d[..., 7]
    flipped_2d[..., 8] = -obs_2d[..., 8]
    
    # 关节镜像 - 展平后批量处理
    flipped_flat = flipped_2d.view(batch_size * 10, 84)
    obs_flat = obs_2d.view(batch_size * 10, 84)
    mirror_joint_tensor(obs_flat, flipped_flat, 9)
    mirror_joint_tensor(obs_flat, flipped_flat, 32)
    mirror_joint_tensor(obs_flat, flipped_flat, 55)
    flipped_2d = flipped_flat.view(batch_size, 10, 84)
    
    # # gait_clock 和其他交换
    flipped_2d[..., 78], flipped_2d[..., 79] = obs_2d[..., 79].clone(), obs_2d[..., 78].clone()
    flipped_2d[..., 80], flipped_2d[..., 81] = obs_2d[..., 81].clone(), obs_2d[..., 80].clone()
    flipped_2d[..., 82], flipped_2d[..., 83] = obs_2d[..., 83].clone(), obs_2d[..., 82].clone()
    
    result[batch_size:] = flipped_2d.view(batch_size, 840)
    return result

def mirror_observation_critic(obs):
    """
    obs: (..., H * 89)  critic 历史观测，单帧 89 维（84 actor + 3 root_lin_vel + 2 feet_contact）
    return: (..., 2 * H * 89)  原观测 + 镜像观测
    """
    if obs is None:
        return obs

    *batch_shape, _ = obs.shape
    batch_size = obs.shape[0] if batch_shape else 1

    per_frame_dim = 89
    history_len = obs.shape[-1] // per_frame_dim
    
    # 预分配输出
    result = torch.empty(batch_size * 2, history_len * per_frame_dim, device=obs.device, dtype=obs.dtype)
    result[:batch_size] = obs
    
    obs_3d = obs.view(batch_size, history_len, per_frame_dim)
    flipped_3d = obs_3d.clone()

    root_offset = 84
    feet_offset = root_offset + 3

    # 向量化镜像所有帧
    flipped_3d[..., 0] = -obs_3d[..., 0]   # base ang vel x
    flipped_3d[..., 2] = -obs_3d[..., 2]   # base ang vel z
    flipped_3d[..., 4] = -obs_3d[..., 4]   # projected gravity y
    flipped_3d[..., 7] = -obs_3d[..., 7]   # command y
    flipped_3d[..., 8] = -obs_3d[..., 8]   # command z
    
    # 关节镜像 - 展平批量处理
    flipped_flat = flipped_3d.view(batch_size * history_len, per_frame_dim)
    obs_flat = obs_3d.view(batch_size * history_len, per_frame_dim)
    mirror_joint_tensor(obs_flat, flipped_flat, 9)
    mirror_joint_tensor(obs_flat, flipped_flat, 32)
    mirror_joint_tensor(obs_flat, flipped_flat, 55)
    flipped_3d = flipped_flat.view(batch_size, history_len, per_frame_dim)
    
    # gait 交换
    flipped_3d[..., 78], flipped_3d[..., 79] = obs_3d[..., 79].clone(), obs_3d[..., 78].clone()
    flipped_3d[..., 80], flipped_3d[..., 81] = obs_3d[..., 81].clone(), obs_3d[..., 80].clone()
    flipped_3d[..., 82], flipped_3d[..., 83] = obs_3d[..., 83].clone(), obs_3d[..., 82].clone()

    # root_lin_vel：y 轴取反
    flipped_3d[..., root_offset + 1] = -obs_3d[..., root_offset + 1]
    
    # feet_contact：左右互换
    flipped_3d[..., feet_offset], flipped_3d[..., feet_offset + 1] = \
        obs_3d[..., feet_offset + 1].clone(), obs_3d[..., feet_offset].clone()

    result[batch_size:] = flipped_3d.view(batch_size, history_len * per_frame_dim)
    return result


def mirror_actions(actions):
    if actions is None:
        return None

    batch_size = actions.shape[0]
    # 预分配输出，避免 vstack
    result = torch.empty(batch_size * 2, actions.shape[-1], device=actions.device, dtype=actions.dtype)
    result[:batch_size] = actions
    
    # 直接在预分配空间中镜像
    mirror_joint_tensor(actions, result[batch_size:], offset=0)
    return result


def data_augmentation_func_g1(env, obs, actions, obs_type):
    
  
    if obs_type == "policy":
        obs_batch = mirror_observation_policy(obs)
    elif obs_type == "critic":
        obs_batch = mirror_observation_critic(obs)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")
    
    mean_actions_batch = mirror_actions(actions)
    return obs_batch, mean_actions_batch

