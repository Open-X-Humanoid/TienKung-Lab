from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    pkg_share = FindPackageShare('tiangong2dex_urdf').find('tiangong2dex_urdf')
    
    # 👇👇👇 修改这里：选择你的 URDF 文件 👇👇👇
    urdf_file = 'tiangong2dex.urdf'  # ← 改成你想用的文件名！
    # urdf_file = 'tiangong2dex.urdf.xacro'  # 如果要用 xacro，取消注释这行
    
    default_model_path = os.path.join(pkg_share, 'urdf', urdf_file)

    declare_model_path = DeclareLaunchArgument(
        name='model',
        default_value=default_model_path,
        description='Absolute path to robot URDF file'
    )

    # 启动 Gazebo（空世界）
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    # 根据文件扩展名决定如何加载
    if urdf_file.endswith('.xacro'):
        robot_description_content = Command(['xacro', ' ', LaunchConfiguration('model')])
    else:
        with open(default_model_path, 'r') as f:
            robot_description_content = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'tiangong2dex',
            '-topic', 'robot_description',
            '-z', '1.5'  # 抬高避免穿地（根据机器人高度调整）
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_model_path,
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])