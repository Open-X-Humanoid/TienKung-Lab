import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from legged_lab.assets import ASSET_DIR
from legged_lab.assets import ISAAC_ASSET_DIR

DEX_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAAC_ASSET_DIR}/tiangong_dex_urdf_EVT2/tiangong_dex_EVT2/urdf/tiangong2dex.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "hip_yaw_l_joint": -0.0,   # l_hip_yaw
            "hip_roll_l_joint": 0.0,    # l_hip_roll
            "hip_pitch_l_joint": -0.15,   # l_hip_pitch
            "knee_pitch_l_joint": 0.3,  # l_knee
            "ankle_pitch_l_joint": -0.15, # l_ankle_pitch
            "ankle_roll_l_joint": 0.0,  # l_ankle_roll
            "hip_yaw_r_joint": -0.0,   # r_hip_yaw
            "hip_roll_r_joint": 0.0,    # r_hip_roll
            "hip_pitch_r_joint": -0.15,   # r_hip_pitch
            "knee_pitch_r_joint": 0.3,  # r_knee
            "ankle_pitch_r_joint": -0.15, # r_ankle_pitch
            "ankle_roll_r_joint": 0.0,  # r_ankle_roll
            "waist_yaw_joint": 0.0,    # waist_roll
            "waist_roll_joint": 0.0,   # waist_pitch
            "waist_pitch_joint": 0.0,     # waist_yaw
            "shoulder_pitch_l_joint": 0.00, # l_shoulder_pitch
            "shoulder_roll_l_joint": 0.10,  # l_shoulder_roll
            "shoulder_yaw_l_joint": 0.0,   # l_shoulder_yaw
            "elbow_pitch_l_joint": -0.50,   # l_elbow
            # "elbow_yaw_l_joint": 0.0,      # l_wrist_yaw
            # "wrist_pitch_l_joint": 0.0,    # l_wrist_pitch
            # "wrist_roll_l_joint": 0.0,     # l_wrist_roll
            "shoulder_pitch_r_joint": 0.00, # r_shoulder_pitch
            "shoulder_roll_r_joint": -0.10, # r_shoulder_roll
            "shoulder_yaw_r_joint": 0.0,   # r_shoulder_yaw
            "elbow_pitch_r_joint": -0.50,   # r_elbow
            # "elbow_yaw_r_joint": 0.0,      # r_wrist_yaw
            # "wrist_pitch_r_joint": 0.0,    # r_wrist_pitch
            # "wrist_roll_r_joint": 0.0      # r_wrist_roll
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip_yaw.*joint",
                ".*hip_roll.*joint",
                ".*hip_pitch.*joint",
                ".*knee_pitch.*joint",
            ],
            effort_limit_sim={
                ".*hip_yaw.*joint": 142,
                ".*hip_roll.*joint": 200,
                ".*hip_pitch.*joint": 200,
                ".*knee_pitch.*joint": 330,
            },
            velocity_limit_sim={
                ".*hip_yaw.*joint": 13.823,
                ".*hip_roll.*joint": 12.566,
                ".*hip_pitch.*joint": 12.566,
                ".*knee_pitch.*joint": 11.106,
            },
            stiffness={
                ".*hip_yaw.*joint": 150,
                ".*hip_roll.*joint": 300,
                ".*hip_pitch.*joint": 300,
                ".*knee_pitch.*joint": 350,
            },  # <-- 你来填
            damping={
                ".*hip_yaw.*joint": 5,
                ".*hip_roll.*joint": 10,
                ".*hip_pitch.*joint": 10,
                ".*knee_pitch.*joint": 10,
            },    # <-- 你来填
        ),

        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*ankle_pitch.*joint",
                ".*ankle_roll.*joint",
            ],
            effort_limit_sim={
                ".*ankle_pitch.*joint": 100,
                ".*ankle_roll.*joint": 50,
            },
            velocity_limit_sim={
                ".*ankle_pitch.*joint": 14.6,
                ".*ankle_roll.*joint": 14.6,
            },
            stiffness={
                ".*ankle_pitch.*joint": 30,
                ".*ankle_roll.*joint": 16.8,
            },  # <-- 你来填
            damping={
                ".*ankle_pitch.*joint": 2.5,
                ".*ankle_roll.*joint": 1.4,
            },    # <-- 你来填
        ),

        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint", 
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 91,
                "waist_roll_joint": 91,
                "waist_pitch_joint": 91,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 9.2153,
                "waist_roll_joint": 9.2153,
                "waist_pitch_joint": 9.2153,
            },
            stiffness={
                "waist_yaw_joint": 400,
                "waist_roll_joint": 400,
                "waist_pitch_joint": 400,
            },  # <-- 你来填
            damping={
                "waist_yaw_joint": 5,
                "waist_roll_joint": 10,
                "waist_pitch_joint": 10,
            },    # <-- 你来填
        ),

        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch.*joint",
                ".*shoulder_roll.*joint",
                ".*shoulder_yaw.*joint",
                ".*elbow_pitch.*joint",
                # ".*elbow_yaw.*joint",
                # ".*wrist_roll.*joint",
                # ".*wrist_pitch.*joint",
            ],
            effort_limit_sim={
                ".*shoulder_pitch.*joint": 90,
                ".*shoulder_roll.*joint": 60,
                ".*shoulder_yaw.*joint": 36,
                ".*elbow_pitch.*joint": 60,
                # ".*elbow_yaw.*joint": 36,
                # ".*wrist_roll.*joint": 36,
                # ".*wrist_pitch.*joint": 36,
            },
            velocity_limit_sim={
                ".*shoulder_pitch.*joint": 15.184,
                ".*shoulder_roll.*joint": 14.137,
                ".*shoulder_yaw.*joint": 9.739,
                ".*elbow_pitch.*joint": 14.137,
                # ".*elbow_yaw.*joint": 9.739,
                # ".*wrist_roll.*joint": 9.739,
                # ".*wrist_pitch.*joint": 9.739,
            },
            stiffness={
                ".*shoulder_pitch.*joint": 150,
                ".*shoulder_roll.*joint": 50,
                ".*shoulder_yaw.*joint": 50,
                ".*elbow_pitch.*joint": 150,
                # ".*elbow_yaw.*joint": 50,
                # ".*wrist_roll.*joint": 20,
                # ".*wrist_pitch.*joint": 20,
            },  # <-- 你来填
            damping={
                ".*shoulder_pitch.*joint": 5,
                ".*shoulder_roll.*joint": 2.5,
                ".*shoulder_yaw.*joint": 2.5,
                ".*elbow_pitch.*joint": 5,
                # ".*elbow_yaw.*joint": 5,
                # ".*wrist_roll.*joint": 2,
                # ".*wrist_pitch.*joint": 2,
            },    # <-- 你来填
        ),
    },
)



D3_ACTION_SCALE = {}
for a in DEX_V3_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            D3_ACTION_SCALE[n] = 0.25  