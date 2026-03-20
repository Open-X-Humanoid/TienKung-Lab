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

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

TIENKUNG2LITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/tienkung2_lite/usd/tienkung2_lite.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.25,
            "hip_yaw_l_joint": 0.0,
            "knee_pitch_l_joint": 0.5,
            "ankle_pitch_l_joint": -0.25,
            "ankle_roll_l_joint": -0.0,
            "hip_roll_r_joint": -0.0,
            "hip_pitch_r_joint": -0.25,
            "hip_yaw_r_joint": 0.0,
            "knee_pitch_r_joint": 0.5,
            "ankle_pitch_r_joint": -0.25,
            "ankle_roll_r_joint": 0.0,
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": -0.0,
            "elbow_pitch_l_joint": -0.5,
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "hip_yaw_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 180,
                "hip_pitch_.*_joint": 300,
                "hip_yaw_.*_joint": 180,
                "knee_pitch_.*_joint": 300,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 15.6,
                "hip_pitch_.*_joint": 15.6,
                "hip_yaw_.*_joint": 15.6,
                "knee_pitch_.*_joint": 15.6,
            },
            stiffness={
                "hip_roll_.*_joint": 700,
                "hip_pitch_.*_joint": 700,
                "hip_yaw_.*_joint": 500,
                "knee_pitch_.*_joint": 700,
            },
            damping={
                "hip_roll_.*_joint": 10,
                "hip_pitch_.*_joint": 10,
                "hip_yaw_.*_joint": 5,
                "knee_pitch_.*_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_pitch_.*_joint",
                "ankle_roll_.*_joint",
            ],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 60,
                "ankle_roll_.*_joint": 30,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 12.8,
                "ankle_roll_.*_joint": 7.8,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30,
                "ankle_roll_.*_joint": 16.8,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
                "ankle_roll_.*_joint": 1.4,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 52.5,
                "shoulder_roll_.*_joint": 52.5,
                "shoulder_yaw_.*_joint": 52.5,
                "elbow_pitch_.*_joint": 52.5,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 14.1,
                "shoulder_roll_.*_joint": 14.1,
                "shoulder_yaw_.*_joint": 14.1,
                "elbow_pitch_.*_joint": 14.1,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60,
                "shoulder_roll_.*_joint": 20,
                "shoulder_yaw_.*_joint": 10,
                "elbow_pitch_.*_joint": 10,
            },
            damping={
                "shoulder_pitch_.*_joint": 3,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1,
                "elbow_pitch_.*_joint": 1,
            },
        ),
    },
)
