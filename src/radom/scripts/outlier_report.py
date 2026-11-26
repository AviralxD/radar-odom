#!/usr/bin/env python3
"""Collects the outlier ablation runs and writes its report after playback.

Writes to <results_dir>/outlier_ablation_<sequence>_<time>/:
  outlier_ablation.png  per level, RANSAC vs GNC: horizontal and yaw RMSE,
                        outliers rejected / all points, inliers rejected / all inliers
  trajectory_pXX.png    both methods against GT at `trajectory_level`
  metrics.csv           every number in the figures
  est_pXX_<method>.tum  aligned estimates; gt.tum for comparison

Alignment as in gt_viz.py: each estimate starts on the first GT position and is
rotated about it by the heading that best fits all positions since. Yaw RMSE
uses the GT orientation, which is reliable on parking_04 (not on stream_02).
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation, Slerp

RESTART_NS = 500_000_000  # a stamp this far in the past means a dataset restart
GT_COLOR = "#212529"
REFERENCE_COLOR = "#868e96"
# Share of the points entering the robust estimator that are perturbed: the
# "outliers rejected [% of points]" of a front-end that rejects every outlier
OUTLIER_SHARE = "outliers in estimator input [% of points]"


class Run:
    """Everything received from one method at one outlier level."""

    def __init__(self):
        self.stamps, self.raw, self.gt = [], [], []  # poses: x y z qx qy qz qw
        # Summed over scans: points entering the robust estimator, and inliers
        self.points = self.inliers = self.injected = self.injected_inliers = 0
        self.scans = 0

    def aligned(self):
        """Aligned positions, horizontal errors [m] and yaw errors [deg]."""
        raw, gt = np.array(self.raw), np.array(self.gt)
        a, b = raw[1:, :2] - raw[0, :2], gt[1:, :2] - gt[0, :2]
        yaw = math.atan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
        r = Rotation.from_euler("z", yaw)
        pos = r.apply(raw[:, :3]) + gt[0, :3] - r.apply(raw[0, :3])
        rot = r * Rotation.from_quat(raw[:, 3:])
        yaw_err = (rot * Rotation.from_quat(gt[:, 3:]).inv()).as_euler("ZYX", degrees=True)[:, 0]
        return pos, rot, np.linalg.norm((pos - gt[:, :3])[:, :2], axis=1), yaw_err

    def metrics(self):
        if len(self.raw) < 2:
            return None
        _, _, horizontal, yaw = self.aligned()
        # Outliers: the perturbed points. Inliers: the untouched ones, which
        # include the scan's own outliers (moving objects), hence the clean runs'
        # few percent of rejected inliers.
        inliers = self.points - self.injected
        outliers_rejected = self.injected - self.injected_inliers
        inliers_rejected = inliers - (self.inliers - self.injected_inliers)

        def share(part, whole):
            return 100.0 * part / whole if whole else math.nan

        return {
            "horizontal RMSE [m]": float(np.sqrt(np.mean(horizontal**2))),
            "final horizontal error [m]": float(horizontal[-1]),
            "yaw RMSE [deg]": float(np.sqrt(np.mean(yaw**2))),
            OUTLIER_SHARE: share(self.injected, self.points),
            "outliers rejected [% of points]": share(outliers_rejected, self.points),
            "outliers rejected [% of outliers]": share(outliers_rejected, self.injected),
            "inliers rejected [% of inliers]": share(inliers_rejected, inliers),
            "scans": self.scans,
        }


class OutlierReport(Node):
    def __init__(self):
        super().__init__("outlier_report")
        gt_file = self.declare_parameter("gt_file", "").value
        self.sequence = self.declare_parameter("sequence_name", "").value
        namespace = self.declare_parameter("output_namespace", "/outlier_ablation").value
        self.config_file = self.declare_parameter("ablation_file", "").value
        self.results_dir = self.declare_parameter("results_dir", "results").value
        self.idle_save_s = self.declare_parameter("idle_save_s", 5.0).value

        with open(self.config_file) as f:
            spec = yaml.safe_load(f)
        self.levels = [0] + [int(v) for v in spec["levels"] if int(v) != 0]
        self.methods = spec["methods"]
        self.traj_level = int(spec.get("trajectory_level", self.levels[-1]))

        raw = np.loadtxt(gt_file, dtype=str)  # stamp_ns x y z qx qy qz qw
        stamps = raw[:, 0].astype(np.int64)
        self.t0_ns = int(stamps[0])
        self.gt_t = (stamps - self.t0_ns) * 1e-9
        self.gt_p = raw[:, 1:4].astype(float)
        self.gt_rot = Slerp(self.gt_t, Rotation.from_quat(raw[:, 4:8].astype(float)))

        for level in self.levels:
            for method in self.methods:
                key = (level, method)
                topic = f"{namespace}/p{level:02d}/{method}"
                self.create_subscription(
                    Odometry, f"{topic}/odom", lambda m, k=key: self.on_odom(k, m), 500)
                self.create_subscription(
                    DiagnosticArray, f"{topic}/stats", lambda m, k=key: self.on_stats(k, m), 500)
        self.reset()
        self.create_timer(1.0, self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"outlier ablation: levels {self.levels[1:]} %, methods {list(self.methods)}")

    def reset(self):
        self.runs = {(level, m): Run() for level in self.levels for m in self.methods}
        self.last_ns = {}
        self.unsaved = False
        self.last_rx = time.monotonic()

    def accept(self, stream, stamp):
        """Time since the first GT pose [s], or None outside the GT."""
        ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if ns < self.last_ns.get(stream, ns) - RESTART_NS:
            self.save()
            self.reset()
        self.last_ns[stream] = ns
        t = (ns - self.t0_ns) * 1e-9
        return t if self.gt_t[0] <= t <= self.gt_t[-1] else None

    def on_odom(self, key, msg):
        t = self.accept((key, "odom"), msg.header.stamp)
        if t is None:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        run = self.runs[key]
        run.stamps.append(t)
        run.raw.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        gt_p = [np.interp(t, self.gt_t, self.gt_p[:, i]) for i in range(3)]
        run.gt.append(np.concatenate([gt_p, self.gt_rot([t])[0].as_quat()]))
        self.unsaved = True
        self.last_rx = time.monotonic()

    def on_stats(self, key, msg):
        if self.accept((key, "stats"), msg.header.stamp) is None or not msg.status:
            return
        values = {kv.key: float(kv.value) for kv in msg.status[0].values}
        if values.get("points", 0) <= 0:  # IMU warm-up: the front-end did not run
            return
        run = self.runs[key]
        run.scans += 1
        run.points += int(values["points"])
        run.inliers += int(values.get("inliers", 0))
        run.injected += int(values.get("injected_points", 0))
        run.injected_inliers += int(values.get("injected_inliers", 0))

    def on_timer(self):
        if self.unsaved and time.monotonic() - self.last_rx > self.idle_save_s:
            self.save()

    # ---------------------------------------------------------------- report

    def save(self):
        if not self.unsaved:
            return
        self.unsaved = False
        table = {key: run.metrics() for key, run in self.runs.items()}
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.expanduser(self.results_dir),
                           f"outlier_ablation_{self.sequence}_{stamp}")
        os.makedirs(out, exist_ok=True)
        shutil.copy(self.config_file, os.path.join(out, "outlier_ablation.yaml"))

        columns = next((list(m) for m in table.values() if m), [])
        with open(os.path.join(out, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["outlier level [%]", "method", *columns])
            for (level, method), m in table.items():
                writer.writerow([level, method, *(
                    "" if m is None else f"{m[c]:.4g}" if isinstance(m[c], float) else m[c]
                    for c in columns)])
        self.write_tum(out)

        notes = self.check_scan_counts()
        self.plot_bars(table, os.path.join(out, "outlier_ablation.png"), notes)
        self.plot_trajectory(table, os.path.join(out, f"trajectory_p{self.traj_level:02d}.png"))
        self.get_logger().info(f"outlier ablation results written to {out}")
        for note in notes:
            self.get_logger().warn(note)

    def write_tum(self, out):
        for (level, method), run in self.runs.items():
            if len(run.raw) < 2:
                continue
            pos, rot, _, _ = run.aligned()
            rows = np.column_stack([np.array(run.stamps) + self.t0_ns * 1e-9, pos, rot.as_quat()])
            np.savetxt(os.path.join(out, f"est_p{level:02d}_{method}.tum"), rows, fmt="%.9f")
        rows = np.column_stack([self.gt_t + self.t0_ns * 1e-9, self.gt_p,
                                self.gt_rot(self.gt_t).as_quat()])
        np.savetxt(os.path.join(out, "gt.tum"), rows, fmt="%.9f")

    def check_scan_counts(self):
        """Every run should have seen every scan; dropped scans make levels unequal."""
        counts = {key: run.scans for key, run in self.runs.items()}
        most = max(counts.values(), default=0)
        short = [f"p{lv:02d}/{m}: {n}" for (lv, m), n in counts.items() if n < 0.99 * most]
        return [f"runs with fewer scans than the others ({most}): {', '.join(short)}"] if short else []

    def plot_bars(self, table, filename, notes):
        # (column, title, unit, reference): "clean" draws each method's result on
        # the clean scans, "ideal" the value of a front-end rejecting every outlier
        panels = [
            ("horizontal RMSE [m]", "horizontal position RMSE", "[m]", "clean"),
            ("yaw RMSE [deg]", "yaw RMSE", "[deg]", "clean"),
            ("outliers rejected [% of points]",
             "outliers rejected / all points", "[%]", "ideal"),
            ("inliers rejected [% of inliers]",
             "inliers rejected / all inliers", "[%]", "clean"),
        ]
        levels = self.levels[1:]
        x = np.arange(len(levels))
        width = 0.8 / len(self.methods)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (column, title, unit, reference) in zip(axes.flat, panels):
            top = 0.0
            for i, (method, spec) in enumerate(self.methods.items()):
                values = np.array([(table[(lv, method)] or {}).get(column, math.nan)
                                   for lv in levels], dtype=float)
                offset = (i - (len(self.methods) - 1) / 2) * width
                bars = ax.bar(x + offset, values, width * 0.92, color=spec["color"], zorder=3)
                ax.bar_label(bars, labels=[_fmt(v) for v in values], fontsize=7.5,
                             padding=2, color="#343a40")
                top = np.nanmax([top, *values])
                clean = (table[(0, method)] or {}).get(column, math.nan)
                if reference == "clean" and math.isfinite(clean):
                    ax.axhline(clean, color=spec["color"], linestyle="--", linewidth=1.2,
                               zorder=2)
                    top = max(top, clean)
            if reference == "ideal":
                first = next(iter(self.methods))
                ideal = [(table[(lv, first)] or {}).get(OUTLIER_SHARE, math.nan) for lv in levels]
                ax.hlines(ideal, x - 0.45, x + 0.45, color=GT_COLOR, linewidth=2, zorder=4)
            if unit == "[%]":
                ax.set_ylim(0, max(108, top * 1.1))
            elif top > 0:
                ax.set_ylim(0, top * 1.15)
            ax.set_xticks(x, [f"{lv} %" for lv in levels])
            ax.set_xlabel("points replaced by outliers (share of each scan)")
            ax.set_title(title)
            ax.set_ylabel(unit)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

        handles = [Patch(color=spec["color"], label=spec.get("label", m))
                   for m, spec in self.methods.items()]
        handles += [Line2D([], [], color=REFERENCE_COLOR, linestyle="--",
                           label="same method on clean scans (0 %)"),
                    Line2D([], [], color=GT_COLOR, linewidth=2,
                           label="every outlier rejected")]
        heading = (f"Outlier ablation, {self.sequence}: radar + IMU + scan-to-submap, "
                   f"RANSAC vs GNC")
        fig.suptitle("\n".join([heading, *notes]), fontsize=13)
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 0.955 - 0.02 * len(notes)))
        fig.tight_layout(rect=(0, 0, 1, 0.93 - 0.02 * len(notes)))
        fig.savefig(filename, dpi=110)
        plt.close(fig)

    def plot_trajectory(self, table, filename):
        level = self.traj_level
        runs = {m: self.runs.get((level, m)) for m in self.methods}
        runs = {m: r for m, r in runs.items() if r is not None and len(r.raw) >= 2}
        if not runs:
            return
        fig, (ax, ax_err) = plt.subplots(
            2, 1, figsize=(9, 11), gridspec_kw={"height_ratios": [3, 1]})
        gt = np.array(next(iter(runs.values())).gt)
        ax.plot(gt[:, 0], gt[:, 1], ":", color=GT_COLOR, linewidth=2, label="ground truth")
        for method, run in runs.items():
            spec = self.methods[method]
            pos, _, horizontal, _ = run.aligned()
            rmse = table[(level, method)]["horizontal RMSE [m]"]
            label = f"{spec.get('label', method)} (RMSE {_fmt(rmse)} m)"
            ax.plot(pos[:, 0], pos[:, 1], color=spec["color"], linewidth=2, label=label)
            ax_err.plot(np.array(run.stamps) - run.stamps[0], horizontal,
                        color=spec["color"], linewidth=2, label=label)
        ax.plot(*gt[0, :2], "o", color=GT_COLOR, markersize=8, label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{self.sequence}, {level} % of points replaced by outliers, "
                     "radar + IMU + scan-to-submap")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(fontsize=9)
        ax_err.set_title("horizontal position error")
        ax_err.set_xlabel("time [s]")
        ax_err.set_ylabel("[m]")
        for a in (ax, ax_err):
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=110)
        plt.close(fig)


def _fmt(value):
    if not math.isfinite(value):
        return "-"
    return f"{value:.2f}" if abs(value) < 10 else f"{value:.1f}" if abs(value) < 100 else f"{value:.0f}"


def main():
    rclpy.init()
    node = OutlierReport()
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
