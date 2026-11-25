#!/usr/bin/env python3
"""Live GT vs estimates visualisation and comparison report for each dataset run.

The runs and comparisons come from the experiments file. Each run is drawn
against ground truth in RViz. After playback a results folder is written with
results.png (all runs), compare_<name>.png (one per comparison), metrics.csv,
metrics_full.csv, per-run data files and a copy of the configuration. The
experiments file chooses the table columns and graphs of each figure.

Alignment uses GT positions only: each estimate starts on the first GT
position, rotated about it by the heading that best fits all positions since.
The GT orientation is not trusted for this: it comes from an INS heading that
can be far off (stream_02). The GT heading is checked against the direction of
travel the radar measures, and the metrics that need it are left out if it fails.
"""

import csv
import datetime
import math
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation, Slerp

FRAME = "world"  # the GT file's local frame
RESTART_NS = 500_000_000  # a stamp this far in the past means a dataset restart
VEL_HALF_WINDOW_S = 0.1  # for differentiating GT positions
DRIFT_SEGMENT_M = 10.0  # distance travelled for the relative error metrics
PATH_SPACING_M = 0.25  # thins the RViz paths ...
PATH_MAX_POSES = 2000  # ... more on long sequences, to bound the publishing cost
PATH_REDRAW_DEG = 0.1  # redraw an RViz path when its alignment has turned this much
ALIGN_AFTER_M = 10.0  # RViz: GT travel before the fitted heading replaces the GT one
HEADING_CHECK_SPEED = 2.0  # [m/s] slower scans say little about the direction of travel
GT_HEADING_LIMIT_DEG = 5.0  # median GT heading error above which the GT orientation is unused
STAT_KEYS = ("points", "inliers", "iterations", "runtime_us", "accepted",
             "vx", "vy", "vz", "match_runtime_us", "matched", "heading_used",
             "correspondences", "match_yaw_deg")
CHECK_KEYS = ("vel_err", "gt_heading_offset_deg")  # added by gt_viz to each scan
HEADING_PANELS = ("yaw_error", "yaw_bars", "velocity_error")  # need the GT orientation

# Short id used in the experiments file -> (column title, key in metrics())
TABLE_METRICS = {
    "horizontal_rmse": ("horizontal\nRMSE [m]", "ATE horizontal RMSE [m]"),
    "horizontal_final": ("final horizontal\nerror [m]", "final horizontal error [m]"),
    "yaw_rmse": ("yaw\nRMSE [deg]", "yaw error RMSE [deg]"),
    "yaw_final": ("final yaw\nerror [deg]", "final yaw error [deg]"),
    "rmse_3d": ("3D\nRMSE [m]", "ATE 3D RMSE [m]"),
    "z_final": ("final z\nerror [m]", "final z error [m]"),
    "drift": ("drift per\n10 m [m]", "drift per 10 m, horizontal [m]"),
    "drift_yaw": ("yaw drift per\n10 m [deg]", "drift per 10 m, yaw [deg]"),
    "velocity_rmse": ("velocity\nRMSE [m/s]", "velocity RMSE [m/s]"),
    "frontend_us": ("front-end\nmedian [us]", "front-end median [us]"),
    "matcher_ms": ("matcher\nmedian [ms]", "matcher median [ms]"),
    "inliers": ("mean\ninliers [%]", "mean inliers [%]"),
}
# Graph name -> (title, y label)
PANELS = {
    "trajectory": ("trajectory (top down, origin-aligned)", "y [m]"),
    "horizontal_error": ("horizontal position error", "[m]"),
    "yaw_error": ("yaw error", "[deg]"),
    "z_error": ("z error", "[m]"),
    "error_3d": ("3D position error", "[m]"),
    "horizontal_bars": ("horizontal error RMSE", "[m]"),
    "yaw_bars": ("yaw error RMSE", "[deg]"),
    "velocity_error": ("radar velocity error vs GT (1 s median)", "[m/s]"),
    "frontend_runtime": ("front-end runtime (1 s median)", "[us], log"),
    "matcher_runtime": ("scan matcher runtime (1 s median)", "[ms]"),
    "inliers": ("inlier ratio (1 s median)", "[%]"),
}
DEFAULT_METRICS = ["horizontal_rmse", "horizontal_final", "yaw_rmse", "yaw_final"]
DEFAULT_PANELS = ["trajectory", "horizontal_error", "yaw_error"]


class Track:
    """Everything received from one run during the current playback."""

    def __init__(self):
        self.stamps = []
        self.raw, self.gt = [], []  # estimated and GT pose per message: x y z qx qy qz qw
        self.stats = []  # dicts, one per radar scan
        self.turn = np.zeros(2)  # sums of cross and dot products, for alignment()
        self.moved = False  # GT has gone ALIGN_AFTER_M from the start
        # RViz: the path is drawn with `view`, (rotation, translation, yaw)
        self.start_view = self.view = None
        self.path = Path()
        self.path.header.frame_id = FRAME
        self.path_raw = []  # the estimated poses in `path`
        self.path_end = None
        # Aligned positions and yaw errors, filled in for the report
        self.pos = self.yaw_err = None

    def add(self, ns, raw, gt):
        if self.stamps:
            a = raw[:2] - self.raw[0][:2]
            b = gt[:2] - self.gt[0][:2]
            self.turn += (a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
            self.moved = self.moved or np.hypot(*b) >= ALIGN_AFTER_M
        self.stamps.append(ns)
        self.raw.append(raw)
        self.gt.append(gt)

    def alignment(self):
        """Rotation about z and translation taking the estimate into the GT frame.

        The first estimated position is put on GT, and the heading is the least
        squares fit of every position since (a rotation about that start point).
        Only GT positions are used: taking the heading from the first GT pose
        would rotate the whole estimate by any error in the GT heading.
        """
        r = Rotation.from_euler("z", math.atan2(*self.turn))
        return r, self.gt[0][:3] - r.apply(self.raw[0][:3])

    def align_all(self):
        """Aligned positions and yaw errors [deg] of every pose."""
        r, offset = self.alignment()
        raw, gt = np.array(self.raw), np.array(self.gt)
        est_rot = r * Rotation.from_quat(raw[:, 3:])
        yaw = (est_rot * Rotation.from_quat(gt[:, 3:]).inv()).as_euler("ZYX", degrees=True)
        return r.apply(raw[:, :3]) + offset, yaw[:, 0]


def rolling_median(values, window=21):
    if len(values) < window:
        return values
    padded = np.pad(values, window // 2, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.nanmedian(view, axis=1)


def column(track, key):
    return np.array([row[key] for row in track.stats])


def percent(mask_values):
    """Share of ones among the finite entries, or None if there are none."""
    finite = mask_values[np.isfinite(mask_values)]
    return 100.0 * finite.mean() if len(finite) else None


def segment_drift(track):
    """RMSE of horizontal translation and heading change over 10 m segments."""
    gt = np.array(track.gt)
    raw = np.array(track.raw)
    travelled = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1))])
    i = np.arange(0, len(gt), 10)
    j = np.searchsorted(travelled, travelled[i] + DRIFT_SEGMENT_M)
    i, j = i[j < len(gt)], j[j < len(gt)]
    if not len(i):
        return float("nan"), float("nan")

    def relative(poses):
        start = Rotation.from_quat(poses[i, 3:7])
        step = start.inv().apply(poses[j, :3] - poses[i, :3])
        turn = (start.inv() * Rotation.from_quat(poses[j, 3:7])).as_euler("ZYX")[:, 0]
        return step, turn

    step_est, turn_est = relative(raw)
    step_gt, turn_gt = relative(gt)
    translation = np.linalg.norm((step_est - step_gt)[:, :2], axis=1)
    turn = np.degrees(np.angle(np.exp(1j * (turn_est - turn_gt))))
    return (float(np.sqrt(np.mean(translation**2))),
            float(np.sqrt(np.mean(turn**2))))


def gt_heading_error(tracks):
    """Median angle [deg] between the GT heading and the direction of travel the
    radar measures, or None if the vehicle hardly moved.

    Both see the same sideslip (e.g. the radar's lever arm in turns), so with a
    good GT this stays within a few degrees (parking_04 1.5, mountain_01 0.5).
    """
    offsets = np.concatenate([column(t, "gt_heading_offset_deg") for t in tracks.values()])
    offsets = offsets[np.isfinite(offsets)]
    return float(np.median(np.abs(offsets))) if len(offsets) >= 20 else None


def metrics(track, heading_ok):
    """Summary numbers for one run: {label: value or None}.

    Without a usable GT orientation, the numbers that need it are None.
    """
    gt = np.array(track.gt)[:, :3]
    error = track.pos - gt
    horizontal = np.linalg.norm(error[:, :2], axis=1)
    yaw = track.yaw_err
    drift_t, drift_yaw = segment_drift(track) if heading_ok else (None, None)

    def if_heading_ok(value):
        return float(value) if heading_ok else None

    inlier_pct = 100.0 * column(track, "inliers") / np.maximum(column(track, "points"), 1)
    match_us = column(track, "match_runtime_us")
    has_matcher = np.isfinite(match_us).any()
    return {
        "ATE horizontal RMSE [m]": float(np.sqrt(np.mean(horizontal**2))),
        "ATE 3D RMSE [m]": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "final horizontal error [m]": float(horizontal[-1]),
        "final z error [m]": float(error[-1, 2]),
        "yaw error RMSE [deg]": if_heading_ok(np.sqrt(np.mean(yaw**2))),
        "final yaw error [deg]": if_heading_ok(yaw[-1]),
        "drift per 10 m, horizontal [m]": drift_t,
        "drift per 10 m, yaw [deg]": drift_yaw,
        "velocity RMSE [m/s]":
            if_heading_ok(np.sqrt(np.nanmean(column(track, "vel_err")**2))),
        "front-end median [us]": float(np.median(column(track, "runtime_us"))),
        "matcher median [ms]":
            float(np.nanmedian(match_us)) / 1000.0 if has_matcher else None,
        "matcher p95 [ms]":
            float(np.nanpercentile(match_us, 95)) / 1000.0 if has_matcher else None,
        "scans matched [%]": percent(column(track, "matched")),
        "matched heading used [%]": percent(column(track, "heading_used")),
        "mean inliers [%]": float(np.mean(inlier_pct)),
        "scans without velocity update": int(np.sum(column(track, "accepted") == 0)),
        "scans": len(track.stats),
    }


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"


class GtViz(Node):
    def __init__(self):
        super().__init__("gt_viz")
        gt_file = self.declare_parameter("gt_file", "").value
        self.gt_file = gt_file
        self.sequence_name = self.declare_parameter("sequence_name", "").value
        namespace = self.declare_parameter("output_namespace", "/radom").value
        experiments_file = self.declare_parameter("experiments_file", "").value
        self.config_files = list(self.declare_parameter("config_files", [""]).value)
        self.results_dir = self.declare_parameter("results_dir", "results").value
        # Wall-clock silence on the odometry topics after which results are saved
        self.idle_save_s = self.declare_parameter("idle_save_s", 5.0).value

        with open(experiments_file) as f:
            spec = yaml.safe_load(f)
        self.runs = spec["runs"]
        self.comparisons = spec.get("comparisons") or {}
        report = spec.get("report") or {}
        self.metrics_shown = report.get("metrics", DEFAULT_METRICS)
        self.panels_shown = report.get("panels", DEFAULT_PANELS)
        # Fail now rather than after a whole playback
        for figure in [report, *self.comparisons.values()]:
            unknown = ([m for m in figure.get("metrics", []) if m not in TABLE_METRICS]
                       + [p for p in figure.get("panels", []) if p not in PANELS])
            if unknown:
                raise ValueError(f"unknown metrics or panels in experiments file: {unknown}")

        # Columns: stamp_ns x y z qx qy qz qw. Stamps are parsed as integers so
        # that nanoseconds are not rounded away by a float conversion.
        raw = np.loadtxt(gt_file, dtype=str)
        stamps = raw[:, 0].astype(np.int64)
        self.t0_ns = int(stamps[0])
        self.gt_t = (stamps - self.t0_ns) * 1e-9
        self.gt_p = raw[:, 1:4].astype(float)
        self.gt_q = raw[:, 4:8].astype(float)
        self.gt_rot = Slerp(self.gt_t, Rotation.from_quat(self.gt_q))
        length = np.linalg.norm(np.diff(self.gt_p, axis=0), axis=1).sum()
        self.path_spacing = max(PATH_SPACING_M, length / PATH_MAX_POSES)
        self.heading_error, self.heading_ok = None, True  # see save()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gt_path_pub = self.create_publisher(Path, "/radom/viz/gt_path", latched)
        self.gt_pose_pub = self.create_publisher(PoseStamped, "/radom/viz/gt_pose", 10)
        self.path_pubs, self.pose_pubs = {}, {}
        for name in self.runs:
            self.path_pubs[name] = self.create_publisher(
                Path, f"/radom/viz/{name}/path", latched)
            self.pose_pubs[name] = self.create_publisher(
                PoseStamped, f"/radom/viz/{name}/pose", 10)
            self.create_subscription(
                Odometry, f"{namespace}/{name}/odom",
                lambda msg, n=name: self.on_odom(n, msg), 500)
            self.create_subscription(
                DiagnosticArray, f"{namespace}/{name}/stats",
                lambda msg, n=name: self.on_stats(n, msg), 500)

        self.reset_run()
        self.publish_gt_path()

        # Steady clock: these must keep ticking when the dataset clock stops
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.5, self.on_timer, clock=steady)
        self.create_timer(5.0, self.log_error, clock=steady)
        self.get_logger().info(
            f"loaded {len(self.gt_t)} GT poses from {gt_file}, "
            f"comparing {', '.join(self.runs)}")

    # ---------------------------------------------------------------- input

    def reset_run(self):
        self.tracks = {name: Track() for name in self.runs}
        self.last_stamp = {}
        self.run_dir = None
        self.unsaved = False
        self.last_rx = time.monotonic()

    def accept_stamp(self, stream, stamp):
        """Return the stamp in ns if GT covers it, handling dataset restarts."""
        ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if ns < self.last_stamp.get(stream, ns) - RESTART_NS:
            self.save()
            self.reset_run()
        self.last_stamp[stream] = ns
        t = (ns - self.t0_ns) * 1e-9
        margin = VEL_HALF_WINDOW_S
        if not self.gt_t[0] + margin <= t <= self.gt_t[-1] - margin:
            # A few samples at the ends are normal; seconds away is the wrong GT
            if t < self.gt_t[0] - 2.0 or t > self.gt_t[-1] + 2.0:
                self.get_logger().warn(
                    f"data is outside the time range of {self.gt_file}: is gt_dir for "
                    "the sequence being played?", throttle_duration_sec=5.0)
            return None
        return ns

    def gt_position(self, t):
        return np.array([np.interp(t, self.gt_t, self.gt_p[:, i]) for i in range(3)])

    def make_pose(self, p, q, stamp=None):
        msg = PoseStamped()
        msg.header.frame_id = FRAME
        if stamp is not None:
            msg.header.stamp = stamp
        pos, ori = msg.pose.position, msg.pose.orientation
        pos.x, pos.y, pos.z = (float(v) for v in p)
        ori.x, ori.y, ori.z, ori.w = (float(v) for v in q)
        return msg

    def publish_gt_path(self):
        path = Path()
        path.header.frame_id = FRAME
        path.poses = [self.make_pose(p, q) for p, q in zip(self.gt_p[::4], self.gt_q[::4])]
        self.gt_path_pub.publish(path)

    def on_odom(self, name, msg):
        ns = self.accept_stamp((name, "odom"), msg.header.stamp)
        if ns is None:
            return
        track = self.tracks[name]
        t = (ns - self.t0_ns) * 1e-9

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        raw = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        gt_p = self.gt_position(t)
        gt_rot = self.gt_rot([t])[0]
        track.add(ns, raw, np.concatenate([gt_p, gt_rot.as_quat()]))

        if track.start_view is None:
            # Shown until the vehicle has moved far enough to fit the heading
            yaw = (gt_rot * Rotation.from_quat(raw[3:]).inv()).as_euler("ZYX")[0]
            r = Rotation.from_euler("z", yaw)
            track.start_view = track.view = (r, gt_p - r.apply(raw[:3]), yaw)
        if track.path_end is None or np.linalg.norm(raw[:3] - track.path_end) > self.path_spacing:
            track.path_raw.append(raw)
            track.path.poses.append(self.aligned_pose(raw, track.view))
            track.path_end = raw[:3]
        self.pose_pubs[name].publish(self.aligned_pose(raw, track.view, msg.header.stamp))
        if name == next(iter(self.runs)):
            self.gt_pose_pub.publish(
                self.make_pose(gt_p, gt_rot.as_quat(), msg.header.stamp))
        self.unsaved = True
        self.last_rx = time.monotonic()

    def aligned_pose(self, raw, view, stamp=None):
        r, offset, _ = view
        return self.make_pose(r.apply(raw[:3]) + offset,
                              (r * Rotation.from_quat(raw[3:])).as_quat(), stamp)

    def redraw_path(self, track):
        """Follow the alignment as it is refined, once it has turned noticeably."""
        if not track.moved:
            return
        r, offset = track.alignment()
        yaw = r.as_euler("ZYX")[0]
        if abs(math.remainder(yaw - track.view[2], 2 * math.pi)) < math.radians(PATH_REDRAW_DEG):
            return
        track.view = (r, offset, yaw)
        raw = np.array(track.path_raw)
        quat = (r * Rotation.from_quat(raw[:, 3:])).as_quat()
        track.path.poses = [self.make_pose(p, q)
                            for p, q in zip(r.apply(raw[:, :3]) + offset, quat)]

    def on_stats(self, name, msg):
        ns = self.accept_stamp((name, "stats"), msg.header.stamp)
        if ns is None or not msg.status:
            return
        t = (ns - self.t0_ns) * 1e-9
        values = {kv.key: float(kv.value) for kv in msg.status[0].values}
        row = {key: values.get(key, math.nan) for key in STAT_KEYS}
        row["stamp"] = ns

        # Radar velocity measurement against differentiated GT, in the radar frame
        h = VEL_HALF_WINDOW_S
        gt_vel = self.gt_rot([t])[0].inv().apply(
            (self.gt_position(t + h) - self.gt_position(t - h)) / (2 * h))
        est_vel = np.array([row["vx"], row["vy"], row["vz"]])
        row["vel_err"] = float(np.linalg.norm(est_vel - gt_vel))
        # GT check: direction of travel in the GT frame against the radar's
        row["gt_heading_offset_deg"] = math.nan
        if min(np.hypot(*gt_vel[:2]), np.hypot(*est_vel[:2])) > HEADING_CHECK_SPEED:
            row["gt_heading_offset_deg"] = math.degrees(math.atan2(
                est_vel[0] * gt_vel[1] - est_vel[1] * gt_vel[0], est_vel[:2].dot(gt_vel[:2])))
        self.tracks[name].stats.append(row)

    # ---------------------------------------------------------------- live

    def on_timer(self):
        if not self.unsaved:
            return
        for name, track in self.tracks.items():
            if track.path.poses:
                self.redraw_path(track)
                self.path_pubs[name].publish(track.path)
        if time.monotonic() - self.last_rx > self.idle_save_s:
            self.save()

    def log_error(self):
        if time.monotonic() - self.last_rx > 1.0:
            return
        parts = []
        for name, track in self.tracks.items():
            if not track.stamps:
                continue
            r, offset, _ = track.view
            raw, gt = track.raw[-1], track.gt[-1]
            error = np.linalg.norm(r.apply(raw[:3]) + offset - gt[:3])
            yaw = ((r * Rotation.from_quat(raw[3:])) * Rotation.from_quat(gt[3:]).inv())
            parts.append(f"{name}: {error:.2f} m, {yaw.as_euler('ZYX', degrees=True)[0]:+.1f} deg")
        if parts:
            self.get_logger().info("position / yaw error | " + " | ".join(parts))

    # ---------------------------------------------------------------- report

    def write_files(self, tracks, table):
        def write_tum(filename, stamps, rows):
            with open(os.path.join(self.run_dir, filename), "w") as f:
                for ns, row in zip(stamps, rows):
                    values = " ".join(f"{v:.9f}" for v in row)
                    f.write(f"{ns // 10**9}.{ns % 10**9:09d} {values}\n")

        longest = max(tracks.values(), key=lambda track: len(track.stamps))
        write_tum("gt.tum", longest.stamps, longest.gt)
        for name, track in tracks.items():
            write_tum(f"est_{name}.tum", track.stamps, track.raw)
            with open(os.path.join(self.run_dir, f"stats_{name}.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("stamp",) + STAT_KEYS + CHECK_KEYS)
                writer.writeheader()
                writer.writerows(track.stats)

        # metrics.csv: the short matrix. metrics_full.csv: everything.
        with open(os.path.join(self.run_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            keys = [TABLE_METRICS[m][1] for m in self.metrics_shown]
            writer.writerow(["run"] + keys)
            for name in tracks:
                writer.writerow([name] + [format_value(table[name][k]) for k in keys])
        with open(os.path.join(self.run_dir, "metrics_full.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            labels = list(table[next(iter(tracks))])
            writer.writerow(["run"] + labels)
            for name in tracks:
                writer.writerow([name] + [format_value(table[name][k]) for k in labels])

        with open(os.path.join(self.run_dir, "gt_source.txt"), "w") as f:
            f.write(self.gt_file + "\n" + self.heading_note() + "\n")
        for path in self.config_files:
            if os.path.isfile(path):
                shutil.copy(path, self.run_dir)

    def plot(self, tracks, table, filename, title, metric_ids, panels):
        """One figure: a table (one row per run) and only the requested graphs."""
        names = list(tracks)
        t_ref = min(track.stamps[0] for track in tracks.values())

        def seconds(stamps):
            return (np.asarray(stamps, dtype=np.int64) - t_ref) * 1e-9

        def style(name):
            run = self.runs[name]
            return {"color": run["color"], "linestyle": run.get("linestyle", "-"),
                    "linewidth": 1.8, "label": run["label"]}

        # Layout: table, then the trajectory across the width, then graph pairs
        graphs = [p for p in panels if p != "trajectory"]
        heights = [0.35 * (len(names) + 2.5)]
        if "trajectory" in panels:
            heights.append(8.0)
        heights += [3.8] * ((len(graphs) + 1) // 2)
        fig = plt.figure(figsize=(15, sum(heights) + 1.0), constrained_layout=True)
        grid = fig.add_gridspec(len(heights), 2, height_ratios=heights)

        summary = fig.add_subplot(grid[0, :])
        summary.axis("off")
        drawn = summary.table(
            cellText=[[format_value(table[n][TABLE_METRICS[m][1]]) for m in metric_ids]
                      for n in names],
            rowLabels=[self.runs[n]["label"] for n in names],
            rowColours=[matplotlib.colors.to_rgba(self.runs[n]["color"], 0.45)
                        for n in names],
            colLabels=[TABLE_METRICS[m][0] for m in metric_ids],
            bbox=[0.22, 0.0, 0.78, 1.0])
        drawn.auto_set_font_size(False)
        drawn.set_fontsize(11)
        for (row, _), cell in drawn.get_celld().items():
            if row == 0:
                cell.set_height(cell.get_height() * 1.8)

        longest = max(tracks.values(), key=lambda track: len(track.stamps))
        gt = np.array(longest.gt)

        def draw(ax, panel):
            if panel in HEADING_PANELS and not self.heading_ok:
                ax.text(0.5, 0.5, "left out: the GT heading is unreliable here\n"
                        f"(median {self.heading_error:.0f} deg off the radar's direction of travel)",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel("")
                return
            if panel == "trajectory":
                ax.plot(gt[:, 0], gt[:, 1], "k:", linewidth=2.0, label="ground truth")
                for name, track in tracks.items():
                    ax.plot(track.pos[:, 0], track.pos[:, 1], **style(name))
                ax.plot(*gt[0, :2], "go", markersize=9, label="start")
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xlabel("x [m]")
                ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
                return
            if panel in ("horizontal_bars", "yaw_bars"):
                key = TABLE_METRICS["horizontal_rmse" if panel == "horizontal_bars"
                                    else "yaw_rmse"][1]
                for i, name in enumerate(names):
                    ax.barh(i, table[name][key], color=self.runs[name]["color"])
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels([self.runs[n]["label"] for n in names], fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel(PANELS[panel][1])
                ax.set_ylabel("")
                return

            for name, track in tracks.items():
                error = track.pos - np.array(track.gt)[:, :3]
                if panel == "horizontal_error":
                    x, y = track.stamps, np.linalg.norm(error[:, :2], axis=1)
                elif panel == "yaw_error":
                    x, y = track.stamps, track.yaw_err
                elif panel == "z_error":
                    x, y = track.stamps, error[:, 2]
                elif panel == "error_3d":
                    x, y = track.stamps, np.linalg.norm(error, axis=1)
                else:
                    x = [row["stamp"] for row in track.stats]
                    if panel == "velocity_error":
                        y = column(track, "vel_err")
                    elif panel == "frontend_runtime":
                        y = column(track, "runtime_us")
                    elif panel == "matcher_runtime":
                        y = column(track, "match_runtime_us") / 1000.0
                        if not np.isfinite(y).any():
                            continue
                    else:
                        y = (100.0 * column(track, "inliers")
                             / np.maximum(column(track, "points"), 1))
                    y = rolling_median(y)
                ax.plot(seconds(x), y, **style(name))
            if panel == "frontend_runtime":
                ax.set_yscale("log")
            ax.set_xlabel("time [s]")

        row = 1
        if "trajectory" in panels:
            cells = [("trajectory", grid[row, :])]
            row += 1
        else:
            cells = []
        cells += [(p, grid[row + i // 2, i % 2]) for i, p in enumerate(graphs)]
        for panel, cell in cells:
            ax = fig.add_subplot(cell)
            heading, ylabel = PANELS[panel]
            ax.set_title(heading)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.4)
            draw(ax, panel)
            # One key for the time series when there is no trajectory to carry it
            if ("trajectory" not in panels and panel == graphs[-2 if len(graphs) > 1 else -1]
                    and ax.get_legend_handles_labels()[0]):
                ax.legend(fontsize=8, ncol=2)

        length = np.linalg.norm(np.diff(gt[:, :3], axis=0), axis=1).sum()
        heading = ("" if self.heading_ok else
                   "\nGT heading unreliable: yaw, velocity and drift metrics left out")
        fig.suptitle(f"{title}  ({self.sequence_name}, GT path {length:.0f} m){heading}",
                     fontsize=16)
        fig.savefig(os.path.join(self.run_dir, filename), dpi=90)
        plt.close(fig)

    def heading_note(self):
        if self.heading_error is None:
            return "GT heading check: not enough driving to check the GT heading"
        verdict = "ok" if self.heading_ok else (
            "UNRELIABLE, so yaw, velocity and drift metrics are left out (positions are fine to use)")
        return (f"GT heading check: median {self.heading_error:.1f} deg between the GT heading and "
                f"the direction of travel the radar measures (limit {GT_HEADING_LIMIT_DEG:.0f}): "
                f"{verdict}")

    def save(self):
        if not self.unsaved:
            return
        tracks = {name: track for name, track in self.tracks.items()
                  if len(track.stamps) >= 2 and track.stats}
        if not tracks:
            return
        missing = [name for name in self.runs if name not in tracks]
        if missing:
            self.get_logger().warn(f"no data from: {', '.join(missing)}")
        if self.run_dir is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            folder = f"{self.sequence_name}_{stamp}" if self.sequence_name else stamp
            self.run_dir = os.path.join(self.results_dir, folder)
            os.makedirs(self.run_dir, exist_ok=True)

        print("[gt_viz] writing results, please wait ...", flush=True)
        for track in tracks.values():
            track.pos, track.yaw_err = track.align_all()
        self.heading_error = gt_heading_error(tracks)
        self.heading_ok = self.heading_error is None or self.heading_error <= GT_HEADING_LIMIT_DEG
        if not self.heading_ok:
            print(f"[gt_viz] WARNING: {self.heading_note()}", flush=True)
        table = {name: metrics(track, self.heading_ok) for name, track in tracks.items()}
        self.write_files(tracks, table)
        self.plot(tracks, table, "results.png", "All runs",
                  self.metrics_shown, self.panels_shown)
        for key, comparison in self.comparisons.items():
            subset = {n: tracks[n] for n in comparison["runs"] if n in tracks}
            if len(subset) >= 2:
                self.plot(subset, table, f"compare_{key}.png",
                          comparison.get("title", key),
                          comparison.get("metrics", self.metrics_shown),
                          comparison.get("panels", self.panels_shown))

        self.unsaved = False
        # print() still works once the ROS context has been shut down
        keys = [TABLE_METRICS[m][1] for m in self.metrics_shown]
        titles = [TABLE_METRICS[m][0].replace("\n", " ") for m in self.metrics_shown]
        print(f"[gt_viz] {'run':<22}" + "".join(f"{t:>28}" for t in titles))
        for name in tracks:
            print(f"[gt_viz] {name:<22}"
                  + "".join(f"{format_value(table[name][k]):>28}" for k in keys))
        print(f"[gt_viz] saved to {self.run_dir}", flush=True)


def main():
    rclpy.init()
    node = GtViz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
