import launch
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
import launch_ros
import os
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(
        package="tiangong2dex_urdf"
    ).find("tiangong2dex_urdf")
    default_model_path = os.path.join(pkg_share, "urdf", "tiangong2dex.urdf.xacro")
    default_rviz_config_path = os.path.join(pkg_share, "config", "display.rviz")

    args = []
    args.append(
        launch.actions.DeclareLaunchArgument(
            name="model",
            default_value=default_model_path,
            description="Absolute path to ginger URDF file",
        )
    )
    args.append(
        launch.actions.DeclareLaunchArgument(
            name="rvizconfig",
            default_value=default_rviz_config_path,
            description="Absolute path to rviz config file",
        )
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            LaunchConfiguration("model"),
        ]
    )
    robot_description_param = {
        "robot_description": launch_ros.parameter_descriptions.ParameterValue(
            robot_description_content, value_type=str
        )
    }

    # # Static TF
    # static_tf = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="static_transform_publisher",
    #     output="log",
    #     arguments=["0.0", "0.0", "1.0", "0.0", "0.0", "0.0", "world", "base_footprint"],
    # )


    robot_state_publisher_node = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description_param],
    )

    joint_state_publisher_node = launch_ros.actions.Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )

    rviz_node = launch_ros.actions.Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rvizconfig")],
    )

    nodes = [
        # static_tf,
        robot_state_publisher_node,
        joint_state_publisher_node,
        rviz_node,
    ]

    return launch.LaunchDescription(args + nodes)
