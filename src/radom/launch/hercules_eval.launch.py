"""Official HeRCULES file player + radom + live GT vs estimate visualisation."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("radom")
    sequence = LaunchConfiguration("sequence")

    return LaunchDescription([
        # Must match the folder selected with "Load" in the player GUI
        DeclareLaunchArgument(
            "sequence",
            default_value=os.path.expanduser("~/radom_ws/dataset/hercules_parking_04")),
        DeclareLaunchArgument(
            "results_dir",
            default_value=os.path.expanduser("~/radom_ws/dataset/results")),
        DeclareLaunchArgument("player", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),

        # Unmodified official player; only the sensors radom needs are enabled
        Node(
            condition=IfCondition(LaunchConfiguration("player")),
            package="hercules_file_player",
            executable="hercules_file_player",
            output="screen",
            parameters=[{
                "enable_aeva": False,
                "enable_continental": True,
                "enable_radarpolar": False,
                "enable_stereo_left": False,
                "enable_stereo_right": False,
            }],
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, "launch", "radom.launch.py"))),

        Node(
            package="radom",
            executable="gt_viz.py",
            name="gt_viz",
            output="screen",
            parameters=[{
                "gt_file": PathJoinSubstitution([sequence, "PR_GT", "Continental_gt.txt"]),
                "results_dir": LaunchConfiguration("results_dir"),
            }],
            # A pip-installed matplotlib in ~/.local can clash with the system numpy
            additional_env={"PYTHONNOUSERSITE": "1"},
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("rviz")),
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "eval.rviz")],
        ),
    ])
