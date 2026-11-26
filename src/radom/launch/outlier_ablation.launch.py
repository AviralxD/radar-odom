"""Outlier ablation: official HeRCULES player + outlier injection + one radom_node per level.

    player -> /continental/points -> outlier_injector -> /outlier_ablation/pXX/points
           -> radom_pXX (RANSAC and GNC runs) -> /outlier_ablation/pXX/<method>/odom
           -> outlier_report (figures and metrics.csv after playback)

All levels run in parallel on the same playback. Levels, methods and the
perturbation that turns points into outliers are set in config/outlier_ablation.yaml.
Estimator settings come from topics.yaml, as in hercules_eval.launch.py.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

NAMESPACE = "/outlier_ablation"
DISPLAY_ONLY = ("label", "color")


def nodes(context):
    def arg(name):
        return os.path.expanduser(LaunchConfiguration(name).perform(context))

    with open(arg("ablation")) as f:
        spec = yaml.safe_load(f)
    levels = [0] + [int(v) for v in spec["levels"] if int(v) != 0]
    if not all(0 <= v <= 100 for v in levels):
        raise RuntimeError(f"outlier levels must be in [0, 100]: {levels}")
    outliers = spec.get("outliers") or {}
    mix = outliers.get("mix") or {}
    if int(spec.get("trajectory_level", levels[-1])) not in levels:
        raise RuntimeError("trajectory_level must be one of the levels")
    methods = spec["methods"]

    gt_dir = os.path.abspath(arg("gt_dir"))
    gt_file = os.path.join(gt_dir, "Continental_gt.txt")
    if not os.path.isfile(gt_file):
        raise RuntimeError(f"no ground truth at {gt_file} (set gt_dir:= or sequence:=)")
    folder = os.path.basename(gt_dir)
    sequence = os.path.basename(os.path.dirname(gt_dir)) if folder == "PR_GT" else folder

    with open(arg("config")) as f:
        defaults = yaml.safe_load(f)["radom"]["ros__parameters"]

    result = [Node(
        package="radom",
        executable="outlier_injector.py",
        name="outlier_injector",
        output="screen",
        parameters=[{
            "radar_topic": defaults["radar_topic"],
            "output_namespace": NAMESPACE,
            "levels": levels,
            "mix": [float(mix.get(kind, 0.0)) for kind in ("multipath", "moving", "clutter")],
            "multipath_factor": float(outliers.get("multipath_factor", 2.0)),
            "moving_objects": [int(v) for v in outliers.get("moving_objects", [1, 3])],
            "object_speed_mps": [float(v) for v in outliers.get("object_speed_mps", [2.0, 10.0])],
            "object_size_m": float(outliers.get("object_size_m", 3.0)),
            # Marks as outliers the points off the ego-velocity by this much
            "inlier_threshold_mps": float(defaults.get("inlier_threshold_mps", 0.15)),
            "seed": int(spec.get("seed", 0)),
            "doppler_field": defaults.get("doppler_field", "v"),
            "rcs_field": defaults.get("rcs_field", "RCS"),
        }],
        additional_env={"PYTHONNOUSERSITE": "1"},
    )]

    # One process per level, so the levels run in parallel
    for level in levels:
        params = dict(defaults)
        params.update({
            "radar_topic": f"{NAMESPACE}/p{level:02d}/points",
            "output_namespace": f"{NAMESPACE}/p{level:02d}",
            "publish_tf": False,
            "runs": list(methods),
            "use_sim_time": True,
        })
        for method, settings in methods.items():
            merged = {**(spec.get("common") or {}), **(settings or {})}
            params.update({f"{method}.{k}": v for k, v in merged.items()
                           if k not in DISPLAY_ONLY})
        result.append(Node(
            package="radom",
            executable="radom_node",
            name=f"radom_p{level:02d}",
            output="screen",
            parameters=[params],
        ))

    result.append(Node(
        package="radom",
        executable="outlier_report.py",
        name="outlier_report",
        output="screen",
        sigterm_timeout="60",  # rendering the figures on Ctrl+C
        parameters=[{
            "gt_file": gt_file,
            "sequence_name": sequence,
            "output_namespace": NAMESPACE,
            "ablation_file": arg("ablation"),
            "results_dir": arg("results_dir"),
        }],
        additional_env={"PYTHONNOUSERSITE": "1"},
    ))
    return result


def generate_launch_description():
    share = get_package_share_directory("radom")
    return LaunchDescription([
        DeclareLaunchArgument(
            "sequence",
            default_value=os.path.expanduser("~/radom_ws/dataset/hercules_parking_04"),
            description="Sequence folder; load the same one in the player GUI"),
        DeclareLaunchArgument(
            "gt_dir",
            default_value=PathJoinSubstitution([LaunchConfiguration("sequence"), "PR_GT"]),
            description="Folder containing Continental_gt.txt"),
        DeclareLaunchArgument(
            "results_dir", default_value=os.path.expanduser("~/radom_ws/dataset/results")),
        DeclareLaunchArgument(
            "config", default_value=os.path.join(share, "config", "topics.yaml")),
        DeclareLaunchArgument(
            "ablation", default_value=os.path.join(share, "config", "outlier_ablation.yaml")),
        DeclareLaunchArgument("player", default_value="true"),

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
        OpaqueFunction(function=nodes),
    ])
