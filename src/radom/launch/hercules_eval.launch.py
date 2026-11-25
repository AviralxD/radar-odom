"""Official HeRCULES file player + radom test matrix + live comparison against GT."""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

PALETTE = ["#e03131", "#1c7ed6", "#ff8c00", "#d6409f", "#9b6bdf", "#0ca678", "#a98600"]


def resolve_experiments(path):
    """Read an experiments file, filling in labels and colours."""
    with open(path) as f:
        spec = yaml.safe_load(f)
    runs = {}
    for index, (name, settings) in enumerate(spec["runs"].items()):
        settings = settings or {}
        runs[name] = dict(
            settings,
            label=settings.get("label", name),
            color=settings.get("color", PALETTE[index % len(PALETTE)]),
        )
    comparisons = spec.get("comparisons") or {}
    for key, comparison in comparisons.items():
        unknown = [n for n in comparison["runs"] if n not in runs]
        if unknown:
            # Commonly a run that was commented out: skip it, keep the rest
            print(f"[hercules_eval] comparison '{key}' names unknown runs {unknown}")
            comparison["runs"] = [n for n in comparison["runs"] if n in runs]
    return {"runs": runs, "comparisons": comparisons, "report": spec.get("report") or {}}


def write_temp(suffix, content):
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        yaml.safe_dump(content, f, sort_keys=False)
        return f.name


def rviz_config(template_path, runs):
    """Add a path and a pose display per run to the RViz template."""
    with open(template_path) as f:
        config = yaml.safe_load(f)
    displays = config["Visualization Manager"]["Displays"]
    for index, (name, run) in enumerate(runs.items()):
        color = "; ".join(str(int(run["color"][i:i + 2], 16)) for i in (1, 3, 5))
        # Earlier runs are drawn wider, so overlapping paths all stay visible
        width = max(0.8 - 0.14 * index, 0.12)
        scale = max(1.3 - 0.15 * index, 0.5)
        displays.append({
            "Class": "rviz_default_plugins/Path", "Name": run["label"], "Enabled": True,
            "Color": color, "Line Style": "Billboards", "Line Width": width,
            "Topic": {"Value": f"/radom/viz/{name}/path", "Depth": 1,
                      "Durability Policy": "Transient Local",
                      "Reliability Policy": "Reliable"},
        })
        displays.append({
            "Class": "rviz_default_plugins/Pose", "Name": f"{run['label']} (now)",
            "Enabled": True, "Color": color, "Shape": "Arrow",
            "Shaft Length": 3.0 * scale, "Shaft Radius": 0.4 * scale,
            "Head Length": 1.2 * scale, "Head Radius": 0.9 * scale,
            "Topic": {"Value": f"/radom/viz/{name}/pose", "Depth": 5},
        })
    return write_temp(".rviz", config)


def ground_truth(context):
    """GT file to evaluate against, and a short name for its sequence."""
    gt_dir = os.path.abspath(os.path.expanduser(
        LaunchConfiguration("gt_dir").perform(context)))
    gt_file = os.path.join(gt_dir, "Continental_gt.txt")  # radar frame
    folder = os.path.basename(gt_dir)
    name = os.path.basename(os.path.dirname(gt_dir)) if folder == "PR_GT" else folder
    return gt_file, name


def check_ground_truth(context):
    gt_file, _ = ground_truth(context)
    if not os.path.isfile(gt_file):
        raise RuntimeError(f"no ground truth at {gt_file} (set gt_dir:= or sequence:=)")
    return []


def visualisation(context):
    gt_file, sequence_name = ground_truth(context)
    share = get_package_share_directory("radom")
    spec = resolve_experiments(LaunchConfiguration("experiments").perform(context))
    return [
        Node(
            package="radom",
            executable="gt_viz.py",
            name="gt_viz",
            output="screen",
            # Rendering the figures on Ctrl+C takes longer than the default 5 s
            sigterm_timeout="60",
            parameters=[{
                "gt_file": gt_file,
                "sequence_name": sequence_name,
                "results_dir": LaunchConfiguration("results_dir"),
                "experiments_file": write_temp(".yaml", spec),
                # Saved next to the results so a run can be reproduced
                "config_files": [
                    LaunchConfiguration("config").perform(context),
                    LaunchConfiguration("experiments").perform(context),
                ],
            }],
            # A pip-installed matplotlib in ~/.local can clash with the system numpy
            additional_env={"PYTHONNOUSERSITE": "1"},
        ),
        Node(
            condition=IfCondition(LaunchConfiguration("rviz")),
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config(os.path.join(share, "rviz", "eval.rviz"),
                                         spec["runs"])],
        ),
    ]


def generate_launch_description():
    share = get_package_share_directory("radom")

    return LaunchDescription([
        # The ground truth must belong to the folder selected with "Load" in the
        # player GUI. Set `sequence`, or `gt_dir` if the GT is kept elsewhere.
        DeclareLaunchArgument(
            "sequence",
            default_value=os.path.expanduser("~/radom_ws/dataset/hercules_parking_04"),
            description="Sequence folder"),
        DeclareLaunchArgument(
            "gt_dir",
            default_value=PathJoinSubstitution([LaunchConfiguration("sequence"), "PR_GT"]),
            description="Folder containing Continental_gt.txt"),
        DeclareLaunchArgument(
            "results_dir",
            default_value=os.path.expanduser("~/radom_ws/dataset/results")),
        DeclareLaunchArgument(
            "config", default_value=os.path.join(share, "config", "topics.yaml")),
        DeclareLaunchArgument(
            "experiments", default_value=os.path.join(share, "config", "experiments.yaml")),
        DeclareLaunchArgument("player", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=check_ground_truth),

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
                os.path.join(share, "launch", "radom.launch.py")),
            launch_arguments={
                "config": LaunchConfiguration("config"),
                "experiments": LaunchConfiguration("experiments"),
            }.items()),

        OpaqueFunction(function=visualisation),
    ])
