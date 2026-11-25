#!/usr/bin/env python3
"""Live GT vs estimates visualisation and comparison report for each run.

Every estimator listed in `estimators` is drawn against ground truth in RViz.
After a run, results.png, metrics.csv and per-estimator data files are saved.
"""

import csv
import datetime
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
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

# RANSAC is drawn wide and GNC thin on top, so both stay visible where they overlap
STYLES = {
    "ransac": {"color": "#d62728", "linewidth": 2.6, "alpha": 0.75},
    "gnc": {"color": "#1f77b4", "linewidth": 1.2, "alpha": 1.0},
}
FALLBACK_STYLE = {"color": "#9467bd", "linewidth": 1.2, "alpha": 1.0}
STAT_KEYS = ("points", "inliers", "iterations", "runtime_us", "accepted",
             "vx", "vy", "vz")


class Track:
    """Everything received from one estimator during the current run."""

    def __init__(self):
        self.align = None  # (rotation, translation) taking estimate into GT frame
        self.stamps, self.raw, self.pos, self.gt = [], [], [], []
        self.pos_err, self.yaw_err = [], []
        self.stats = []  # dicts, one per radar scan
        self.path = Path()
        self.path.header.frame_id = FRAME


def rolling_median(values, window=21):
    if len(values) < window:
        return values
    padded = np.pad(values, window // 2, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.nanmedian(view, axis=1)


class GtViz(Node):
    def __init__(self):
        super().__init__("gt_viz")
        gt_file = self.declare_parameter("gt_file", "").value
        namespace = self.declare_parameter("output_namespace", "/radom").value
        self.names = list(self.declare_parameter("estimators", ["ransac", "gnc"]).value)
        self.results_dir = self.declare_parameter("results_dir", "results").value
        # Wall-clock silence on the odometry topics after which the run is saved
        self.idle_save_s = self.declare_parameter("idle_save_s", 5.0).value

        # Columns: stamp_ns x y z qx qy qz qw. Stamps are parsed as integers so
        # that nanoseconds are not rounded away by a float conversion.
        raw = np.loadtxt(gt_file, dtype=str)
        stamps = raw[:, 0].astype(np.int64)
        self.t0_ns = int(stamps[0])
        self.gt_t = (stamps - self.t0_ns) * 1e-9
        self.gt_p = raw[:, 1:4].astype(float)
        self.gt_q = raw[:, 4:8].astype(float)
        self.gt_rot = Slerp(self.gt_t, Rotation.from_quat(self.gt_q))

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gt_path_pub = self.create_publisher(Path, "/radom/viz/gt_path", latched)
        self.gt_pose_pub = self.create_publisher(PoseStamped, "/radom/viz/gt_pose", 10)
        self.path_pubs, self.pose_pubs = {}, {}
        for name in self.names:
            self.path_pubs[name] = self.create_publisher(
                Path, f"/radom/viz/{name}/path", latched)
            self.pose_pubs[name] = self.create_publisher(
                PoseStamped, f"/radom/viz/{name}/pose", 10)
            self.create_subscription(
                Odometry, f"{namespace}/{name}/odom",
                lambda msg, n=name: self.on_odom(n, msg), 200)
            self.create_subscription(
                DiagnosticArray, f"{namespace}/{name}/stats",
                lambda msg, n=name: self.on_stats(n, msg), 200)

        self.reset_run()
        self.publish_gt_path()

        # Steady clock: these must keep ticking when the dataset clock stops
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.2, self.on_timer, clock=steady)
        self.create_timer(5.0, self.log_error, clock=steady)
        self.get_logger().info(
            f"loaded {len(self.gt_t)} GT poses from {gt_file}, "
            f"comparing {', '.join(self.names)}")

    # ---------------------------------------------------------------- input

    def reset_run(self):
        self.tracks = {name: Track() for name in self.names}
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
        p = np.array([p.x, p.y, p.z])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        gt_p = self.gt_position(t)
        gt_rot = self.gt_rot([t])[0]

        # Origin alignment (same idea as evo --align_origin): make the first
        # estimated pose coincide with GT, then never adjust again.
        if track.align is None:
            r = gt_rot * rot.inv()
            track.align = (r, gt_p - r.apply(p))
        r, offset = track.align
        est_p = r.apply(p) + offset
        est_rot = r * rot

        track.stamps.append(ns)
        track.raw.append(np.concatenate([p, rot.as_quat()]))
        track.pos.append(est_p)
        track.gt.append(np.concatenate([gt_p, gt_rot.as_quat()]))
        track.pos_err.append(float(np.linalg.norm(est_p - gt_p)))
        track.yaw_err.append(
            float((est_rot * gt_rot.inv()).as_euler("ZYX", degrees=True)[0]))

        pose = self.make_pose(est_p, est_rot.as_quat(), msg.header.stamp)
        track.path.poses.append(pose)
        self.pose_pubs[name].publish(pose)
        if name == self.names[0]:
            self.gt_pose_pub.publish(
                self.make_pose(gt_p, gt_rot.as_quat(), msg.header.stamp))
        self.unsaved = True
        self.last_rx = time.monotonic()

    def on_stats(self, name, msg):
        ns = self.accept_stamp((name, "stats"), msg.header.stamp)
        if ns is None or not msg.status:
            return
        t = (ns - self.t0_ns) * 1e-9
        values = {kv.key: float(kv.value) for kv in msg.status[0].values}
        row = {key: values.get(key, np.nan) for key in STAT_KEYS}
        row["stamp"] = ns

        # Radar velocity measurement against differentiated GT, in the radar frame
        h = VEL_HALF_WINDOW_S
        gt_vel = self.gt_rot([t])[0].inv().apply(
            (self.gt_position(t + h) - self.gt_position(t - h)) / (2 * h))
        est_vel = np.array([row["vx"], row["vy"], row["vz"]])
        row["vel_err"] = float(np.linalg.norm(est_vel - gt_vel))
        self.tracks[name].stats.append(row)

    # ---------------------------------------------------------------- live

    def on_timer(self):
        if not self.unsaved:
            return
        for name, track in self.tracks.items():
            if track.path.poses:
                self.path_pubs[name].publish(track.path)
        if time.monotonic() - self.last_rx > self.idle_save_s:
            self.save()

    def log_error(self):
        if time.monotonic() - self.last_rx > 1.0:
            return
        parts = [
            f"{name}: {track.pos_err[-1]:.2f} m, {track.yaw_err[-1]:+.1f} deg"
            for name, track in self.tracks.items() if track.pos_err
        ]
        if parts:
            self.get_logger().info("position / yaw error | " + " | ".join(parts))

    # ---------------------------------------------------------------- report

    def metrics(self, track):
        """Summary numbers for one estimator, as (label, formatted value) pairs."""
        pos_err = np.array(track.pos_err)
        yaw_err = np.array(track.yaw_err)
        column = {key: np.array([row[key] for row in track.stats])
                  for key in STAT_KEYS + ("vel_err",)}
        z_err = track.pos[-1][2] - track.gt[-1][2]
        inlier_pct = 100.0 * column["inliers"] / np.maximum(column["points"], 1)
        return [
            ("ATE RMSE [m]", f"{np.sqrt(np.mean(pos_err**2)):.3f}"),
            ("final position error [m]", f"{pos_err[-1]:.3f}"),
            ("yaw error RMSE [deg]", f"{np.sqrt(np.mean(yaw_err**2)):.3f}"),
            ("final yaw error [deg]", f"{yaw_err[-1]:+.3f}"),
            ("final z error [m]", f"{z_err:+.3f}"),
            ("velocity RMSE [m/s]",
             f"{np.sqrt(np.nanmean(column['vel_err']**2)):.4f}"),
            ("runtime median [us]", f"{np.median(column['runtime_us']):.1f}"),
            ("runtime p95 [us]", f"{np.percentile(column['runtime_us'], 95):.1f}"),
            ("runtime max [us]", f"{np.max(column['runtime_us']):.1f}"),
            ("mean inliers [%]", f"{np.mean(inlier_pct):.2f}"),
            ("mean iterations *", f"{np.mean(column['iterations']):.1f}"),
            ("scans not used by filter", f"{int(np.sum(column['accepted'] == 0))}"),
            ("scans", f"{len(track.stats)}"),
        ]

    def write_files(self, tracks):
        def stamp_text(ns):
            return f"{ns // 10**9}.{ns % 10**9:09d}"

        def write_tum(path, stamps, rows):
            with open(path, "w") as f:
                for ns, row in zip(stamps, rows):
                    f.write(stamp_text(ns) + " " + " ".join(f"{v:.9f}" for v in row) + "\n")

        longest = max(tracks.values(), key=lambda track: len(track.stamps))
        write_tum(os.path.join(self.run_dir, "gt.tum"), longest.stamps, longest.gt)
        for name, track in tracks.items():
            write_tum(os.path.join(self.run_dir, f"est_{name}.tum"),
                      track.stamps, track.raw)
            with open(os.path.join(self.run_dir, f"stats_{name}.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("stamp",) + STAT_KEYS + ("vel_err",))
                writer.writeheader()
                writer.writerows(track.stats)

    def plot(self, tracks, table):
        t_ref = min(track.stamps[0] for track in tracks.values())

        def seconds(stamps):
            return (np.asarray(stamps, dtype=np.int64) - t_ref) * 1e-9

        def style(name):
            return STYLES.get(name, FALLBACK_STYLE)

        fig = plt.figure(figsize=(18, 18))
        grid = fig.add_gridspec(4, 3, height_ratios=[2.6, 1, 1, 1])

        def panel(row, col, title, ylabel):
            ax = fig.add_subplot(grid[row, col])
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("time [s]")
            ax.grid(True, alpha=0.4)
            return ax

        trajectory = fig.add_subplot(grid[0, :2])
        longest = max(tracks.values(), key=lambda track: len(track.stamps))
        gt = np.array(longest.gt)
        trajectory.plot(gt[:, 0], gt[:, 1], "k--", linewidth=1.4, label="ground truth")
        for name, track in tracks.items():
            pos = np.array(track.pos)
            trajectory.plot(pos[:, 0], pos[:, 1], label=name.upper(), **style(name))
        trajectory.plot(*gt[0, :2], "go", markersize=9, label="start")
        trajectory.set_aspect("equal", adjustable="datalim")
        trajectory.set_title("trajectory (top down, origin-aligned)")
        trajectory.set_xlabel("x [m]")
        trajectory.set_ylabel("y [m]")
        trajectory.grid(True, alpha=0.4)
        trajectory.legend()

        summary = fig.add_subplot(grid[0, 2])
        summary.axis("off")
        names = list(tracks)
        labels = [label for label, _ in table[names[0]]]
        cells = [[table[name][i][1] for name in names] for i in range(len(labels))]
        drawn = summary.table(
            cellText=cells, rowLabels=labels, colLabels=[n.upper() for n in names],
            colColours=[matplotlib.colors.to_rgba(style(n)["color"], 0.35) for n in names],
            bbox=[0.34, 0.12, 0.66, 0.82])
        drawn.auto_set_font_size(False)
        drawn.set_fontsize(10)
        summary.set_title("summary")
        summary.text(0.0, 0.04, "* not comparable: RANSAC hypotheses vs GNC weighted solves",
                     fontsize=9, transform=summary.transAxes)

        position = panel(1, 0, "position error", "[m]")
        yaw = panel(1, 1, "yaw error", "[deg]")
        height = panel(1, 2, "z error", "[m]")
        velocity = panel(2, 0, "radar velocity error vs GT (1 s median)", "[m/s]")
        runtime = panel(3, 0, "front-end runtime (1 s median)", "[us], log")
        runtime.set_yscale("log")
        inliers = panel(3, 2, "inlier ratio (1 s median)", "[%]")
        histogram = fig.add_subplot(grid[3, 1])
        histogram.set_title("front-end runtime distribution")
        histogram.set_xlabel("[us], log")
        histogram.set_ylabel("scans")
        histogram.set_xscale("log")
        histogram.grid(True, alpha=0.4)

        all_runtimes = np.concatenate(
            [[row["runtime_us"] for row in track.stats] for track in tracks.values()])
        bins = np.geomspace(max(all_runtimes.min(), 0.1), all_runtimes.max() * 1.01, 50)

        for name, track in tracks.items():
            kwargs = dict(style(name), label=name.upper())
            t = seconds(track.stamps)
            position.plot(t, track.pos_err, **kwargs)
            yaw.plot(t, track.yaw_err, **kwargs)
            height.plot(t, np.array(track.pos)[:, 2] - np.array(track.gt)[:, 2], **kwargs)

            ts = seconds([row["stamp"] for row in track.stats])
            column = {key: np.array([row[key] for row in track.stats])
                      for key in ("vel_err", "runtime_us", "inliers", "points")}
            velocity.plot(ts, rolling_median(column["vel_err"]), **kwargs)
            runtime.plot(ts, rolling_median(column["runtime_us"]), **kwargs)
            inliers.plot(ts, rolling_median(
                100.0 * column["inliers"] / np.maximum(column["points"], 1)), **kwargs)
            histogram.hist(column["runtime_us"], bins=bins, histtype="step",
                           color=style(name)["color"], linewidth=1.6, label=name.upper())

        # Direct differences: the error curves often overlap too closely to compare
        separation = panel(2, 1, "distance between the two estimates", "[m]")
        measurement = panel(2, 2, "difference between velocity measurements", "[m/s]")
        if len(names) >= 2:
            a, b = tracks[names[0]], tracks[names[1]]
            common, ia, ib = np.intersect1d(a.stamps, b.stamps, return_indices=True)
            if len(common):
                gap = np.linalg.norm(np.array(a.pos)[ia] - np.array(b.pos)[ib], axis=1)
                separation.plot(seconds(common), gap, color="k")
            va = {row["stamp"]: (row["vx"], row["vy"], row["vz"]) for row in a.stats}
            vb = {row["stamp"]: (row["vx"], row["vy"], row["vz"]) for row in b.stats}
            shared = sorted(set(va) & set(vb))
            if shared:
                diff = [np.linalg.norm(np.subtract(va[s], vb[s])) for s in shared]
                measurement.plot(seconds(shared), diff, color="k", linewidth=0.8)
        for ax in (position, yaw, height, velocity, runtime, inliers, histogram):
            ax.legend(fontsize=9)

        length = np.linalg.norm(np.diff(gt[:, :3], axis=0), axis=1).sum()
        fig.suptitle(f"{' vs '.join(n.upper() for n in names)} radar-inertial odometry, "
                     f"GT path {length:.0f} m", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(os.path.join(self.run_dir, "results.png"), dpi=100)
        plt.close(fig)

    def save(self):
        if not self.unsaved:
            return
        tracks = {name: track for name, track in self.tracks.items()
                  if len(track.stamps) >= 2 and track.stats}
        if not tracks:
            return
        if self.run_dir is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = os.path.join(self.results_dir, stamp)
            os.makedirs(self.run_dir, exist_ok=True)

        table = {name: self.metrics(track) for name, track in tracks.items()}
        names = list(tracks)
        rows = [[label] + [table[n][i][1] for n in names]
                for i, (label, _) in enumerate(table[names[0]])]
        with open(os.path.join(self.run_dir, "metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + names)
            writer.writerows(rows)
        self.write_files(tracks)
        self.plot(tracks, table)

        self.unsaved = False
        # print() still works once the ROS context has been shut down
        print(f"[gt_viz] {'metric':<28}" + "".join(f"{n:>12}" for n in names))
        for row in rows:
            print(f"[gt_viz] {row[0]:<28}" + "".join(f"{v:>12}" for v in row[1:]))
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
