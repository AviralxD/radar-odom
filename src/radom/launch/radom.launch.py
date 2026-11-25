"""radom_node with the runs defined in an experiments file."""

import os
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DISPLAY_ONLY = ("label", "color", "linestyle")  # used by gt_viz and RViz, not the estimator


def run_parameters(path):
    """Turn the `runs` of an experiments file into radom_node parameters."""
    with open(path) as f:
        spec = yaml.safe_load(f)
    runs = spec.get("runs") or {}
    if not runs:
        raise RuntimeError(f"{path}: `runs` is empty")

    params = {"runs": list(runs)}
    for name, settings in runs.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise RuntimeError(f"{path}: invalid run name '{name}'")
        for key, value in (settings or {}).items():
            if key not in DISPLAY_ONLY:
                params[f"{name}.{key}"] = value

    params["tf_run"] = spec.get("tf_run", next(iter(runs)))
    if params["tf_run"] not in runs:
        raise RuntimeError(f"{path}: tf_run '{params['tf_run']}' is not a run")
    return params


def radom_node(context):
    experiments = LaunchConfiguration("experiments").perform(context)
    return [Node(
        package="radom",
        executable="radom_node",
        name="radom",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            run_parameters(experiments),
            {"use_sim_time": True},
        ],
    )]


def generate_launch_description():
    share = get_package_share_directory("radom")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config", default_value=os.path.join(share, "config", "topics.yaml")),
        DeclareLaunchArgument(
            "experiments", default_value=os.path.join(share, "config", "experiments.yaml")),
        OpaqueFunction(function=radom_node),
    ])
