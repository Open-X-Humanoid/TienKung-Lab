# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.tiangong_dex_urdf_EVT2 import DEX_V3_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.6
    gait_air_ratio_r: float = 0.6
    gait_phase_offset_l: float = 0.6
    gait_phase_offset_r: float = 0.1
    gait_cycle: float = 0.64

@configclass
class DexEventCfg(EventCfg):
    """TienKung 环境的事件配置"""
    
    # 继承所有父类属性，并添加新的
    randomize_pd_gains: EventTerm = None
    randomize_apply_external_force_torque:EventTerm = None
    randomize_rigid_body_com:EventTerm = None

@configclass
class DexRewardCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"std": 0.5})  # 2.5 -> 4.0 增强
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})  # 2.5 -> 2.0 略降
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    torso_ang_vel_xy_l2 = RewTerm(func=mdp.body_ang_vel_xy_l2, 
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
                                  weight=-0.5)
    torso_ang_acc_xy_l2 = RewTerm(func=mdp.body_ang_acc_xy_l2, 
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")},
                                  weight=-1e-4)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,  # -1.0 -> -0.5 降低
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 5.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-0.25  # -0.5 -> -0.25
    )
    waist_orientation_l2 = RewTerm(
    func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link")}, weight=-1.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)  # -50 -> -20 降低过度惩罚
    alive_reward = RewTerm(func=mdp.alive_reward, weight=0.5)  # 新增：持续存活奖励
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-0.5,  # -2.0 -> -0.5 降低
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,  # -2.0 -> -0.5 降低
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)  # 函数不存在，注释掉
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"])},
    )
    joint_deviation_waist = RewTerm(    
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,  # -0.02 -> -0.05: slightly increase to have more effect during walking
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_pitch_.*_joint",
                    "knee_pitch_.*_joint",
                    "ankle_pitch_.*_joint",
                    "ankle_roll_.*_joint",
                ],
            )
        },
    )

    gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=2.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=2.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=1.5, params={"delta_t": 0.02})

    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-0.3)  # -1.0 -> -0.3
    hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-0.3)  # -1.0 -> -0.3
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-0.5)  # -2.0 -> -0.5


@configclass
class DexRunFlatEnvCfg:
    amp_motion_files_display = ["legged_lab/envs/dex/datasets/motion_visualization/run.txt"]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=DEX_V3_CFG,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    reward = DexRewardCfg()
    gait = GaitCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.4,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=False,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=DexEventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
            ),
            randomize_pd_gains=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="reset",  # apply after each scene.reset so default PD isn't restored
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stiffness_distribution_params": (0.75, 1.25),
                    "damping_distribution_params": (0.75, 1.25),
                    "operation": "scale",
                    "distribution": "uniform"
                },
            ),
            randomize_apply_external_force_torque = EventTerm(
                func=mdp.apply_external_force_torque,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "force_range": (-20.0, 20.0),
                    "torque_range": (-5.0, 5.0),
                },
            ),
            randomize_rigid_body_com = EventTerm(
                func=mdp.randomize_rigid_body_com,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "waist_yaw_link"]),
                    "com_range": {
                    "x": (-0.05, 0.05),  
                    "y": (-0.05, 0.05),  
                    "z": (0.0, 0.0),  
                    },
                },
            )
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class DexRunAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=50.0,  
            data_augmentation_func=mdp.data_augmentation_func_g1,
        ),
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "run"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "run"
    wandb_project = "run"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/dex/datasets/motion_amp_expert/new_run.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 23
