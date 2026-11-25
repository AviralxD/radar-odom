#!/usr/bin/env python3
"""Live GT vs estimate visualisation, with a plot and TUM files saved per run."""

import datetime
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation, Slerp

FRAME = "world"  # the GT file's local frame


class GtViz(Node):
    def __init__(self):
        super().__init__("gt_viz")
        gt_file = self.declare_parameter("gt_file", "").value
        odom_topic = self.declare_parameter("odom_topic", "/radom/odom").value
        self.results_dir = self.declare_parameter("results_dir", "results").value
        # Wall-clock silence on the odometry topic after which the run is saved
        self.idle_save_s = self.declare_parameter("idle_save_s", 5.0).value

        # Columns: stamp_ns x y z qx qy qz qw. Stamps are parsed as integers so
        # that nanoseconds are not rounded away by a float conversion.
        raw = np.loadtxt(gt_file, dtype=str)
        stamps = raw[:, 0].astype(np.int64)
        self.t0_ns = stamps[0]
        self.gt_t = (stamps - self.t0_ns) * 1e-9
        self.gt_p = raw[:, 1:4].astype(float)
        self.gt_q = raw[:, 4:8].astype(float)
        self.gt_rot = Slerp(self.gt_t, Rotation.from_quat(self.gt_q))

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gt_path_pub = self.create_publisher(Path, "/radom/viz/gt_path", latched)
        self.est_path_pub = self.create_publisher(Path, "/radom/viz/est_path", latched)
        self.gt_pose_pub = self.create_publisher(PoseStamped, "/radom/viz/gt_pose", 10)
        self.est_pose_pub = self.create_publisher(PoseStamped, "/radom/viz/est_pose", 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 100)

        self.reset_run()
        self.publish_gt_path()

        # Steady clock: these must keep ticking when the dataset clock stops
        steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.2, self.on_timer, clock=steady)
        self.create_timer(5.0, self.log_error, clock=steady)
        self.get_logger().info(f"loaded {len(self.gt_t)} GT poses from {gt_file}")

    def reset_run(self):
        self.align = None  # (rotation, translation) taking estimate into GT frame
        self.t, self.raw_p, self.raw_q = [], [], []
        self.est_p, self.pos_err, self.yaw_err = [], [], []
        self.gt_matched = []
        self.est_path = Path()
        self.est_path.header.frame_id = FRAME
        self.run_dir = None
        self.unsaved = False
        self.last_rx = time.monotonic()

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

    def on_odom(self, msg):
        stamp = msg.header.stamp
        t = (stamp.sec * 1_000_000_000 + stamp.nanosec - int(self.t0_ns)) * 1e-9
        if self.t and t < self.t[-1]:  # dataset restarted
            self.save()
            self.reset_run()
        if not self.gt_t[0] <= t <= self.gt_t[-1]:
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        p = np.array([p.x, p.y, p.z])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        gt_p = np.array([np.interp(t, self.gt_t, self.gt_p[:, i]) for i in range(3)])
        gt_rot = self.gt_rot([t])[0]

        # Origin alignment (same idea as evo --align_origin): make the first
        # estimated pose coincide with GT, then never adjust again.
        if self.align is None:
            r = gt_rot * rot.inv()
            self.align = (r, gt_p - r.apply(p))
        r, offset = self.align
        est_p = r.apply(p) + offset
        est_rot = r * rot

        yaw = (est_rot * gt_rot.inv()).as_euler("ZYX", degrees=True)[0]
        self.t.append(t)
        self.raw_p.append(p)
        self.raw_q.append(rot.as_quat())
        self.est_p.append(est_p)
        self.gt_matched.append((gt_p, gt_rot.as_quat()))
        self.pos_err.append(float(np.linalg.norm(est_p - gt_p)))
        self.yaw_err.append(float(yaw))

        pose = self.make_pose(est_p, est_rot.as_quat(), stamp)
        self.est_path.poses.append(pose)
        self.est_pose_pub.publish(pose)
        self.gt_pose_pub.publish(self.make_pose(gt_p, gt_rot.as_quat(), stamp))
        self.unsaved = True
        self.last_rx = time.monotonic()

    def on_timer(self):
        if not self.unsaved:
            return
        self.est_path_pub.publish(self.est_path)
        if time.monotonic() - self.last_rx > self.idle_save_s:
            self.save()

    def log_error(self):
        if self.t and time.monotonic() - self.last_rx < 1.0:
            self.get_logger().info(
                f"position error {self.pos_err[-1]:.2f} m, "
                f"yaw error {self.yaw_err[-1]:+.1f} deg"
            )

    def save(self):
        if not self.unsaved or len(self.t) < 2:
            return
        if self.run_dir is None:
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.run_dir = os.path.join(self.results_dir, name)
            os.makedirs(self.run_dir, exist_ok=True)

        t_abs = [(int(self.t0_ns) + round(t * 1e9)) for t in self.t]
        gt_rows = [np.concatenate(pair) for pair in self.gt_matched]
        est_rows = [np.concatenate(pair) for pair in zip(self.raw_p, self.raw_q)]
        for fname, rows in (("est.tum", est_rows), ("gt.tum", gt_rows)):
            with open(os.path.join(self.run_dir, fname), "w") as f:
                for ns, row in zip(t_abs, rows):
                    values = " ".join(f"{v:.9f}" for v in row)
                    f.write(f"{ns // 10**9}.{ns % 10**9:09d} {values}\n")

        t = np.array(self.t) - self.t[0]
        est = np.array(self.est_p)
        gt = np.array([pair[0] for pair in self.gt_matched])
        err = np.array(self.pos_err)
        length = np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()
        summary = (
            f"ATE RMSE {np.sqrt(np.mean(err**2)):.2f} m | final {err[-1]:.2f} m | "
            f"final yaw {self.yaw_err[-1]:+.1f} deg | GT path {length:.0f} m"
        )

        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
        ax = fig.add_subplot(grid[:, 0])
        ax.plot(gt[:, 0], gt[:, 1], "k--", label="ground truth")
        ax.plot(est[:, 0], est[:, 1], "r", label="estimate")
        ax.plot(*gt[0, :2], "go", label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        series = (
            ("position error [m]", err),
            ("yaw error [deg]", np.array(self.yaw_err)),
            ("z error [m]", est[:, 2] - gt[:, 2]),
        )
        for row, (label, values) in enumerate(series):
            sub = fig.add_subplot(grid[row, 1])
            sub.plot(t, values)
            sub.set_ylabel(label)
            sub.grid(True)
        sub.set_xlabel("time [s]")
        fig.suptitle(f"{summary}  (origin-aligned)")
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "gt_vs_est.png"), dpi=120)
        plt.close(fig)

        self.unsaved = False
        # print() still works once the ROS context has been shut down
        print(f"[gt_viz] {summary}\n[gt_viz] saved to {self.run_dir}", flush=True)


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
