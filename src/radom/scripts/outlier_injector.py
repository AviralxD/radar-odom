#!/usr/bin/env python3
"""Republishes every radar scan once per outlier level, with points turned into outliers.

<output_namespace>/pXX/points carries the scan with XX % of its points replaced
(p00: unchanged); the number of points stays the same. Replaced points become
one of three kinds of radar outlier (config/outlier_ablation.yaml):

  multipath  double-bounce ghost: same direction, range and Doppler scaled by
             the bounce factor (2: radar -> target -> vehicle -> target -> radar).
             All ghosts agree with each other: they fit `factor` x the true velocity.
  moving     points of 1-3 objects, each a cluster in azimuth and range, whose
             Doppler includes the object's own velocity. Each object fits its own
             wrong ego-velocity.
  clutter    random direction, range and Doppler within the scan's spread.

Multipath and moving objects are coherent, so they compete with the real
static points for the consensus instead of scattering like noise.

The `injected` field marks the replaced points that are Doppler outliers: those
off the scan's ego-velocity by more than the inlier threshold. The ego-velocity
is fitted to the clean scan here. A ghost of a static point seen while standing
still is not a Doppler outlier, so it is left unmarked.
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField

NUMPY_TYPES = {
    PointField.INT8: "i1", PointField.UINT8: "u1", PointField.INT16: "<i2",
    PointField.UINT16: "<u2", PointField.INT32: "<i4", PointField.UINT32: "<u4",
    PointField.FLOAT32: "<f4", PointField.FLOAT64: "<f8",
}
OUT_NAMES = ["x", "y", "z", "v", "RCS", "injected"]
OUT_DTYPE = np.dtype({"names": OUT_NAMES, "formats": ["<f4"] * 5 + ["u1"],
                      "offsets": [0, 4, 8, 12, 16, 20], "itemsize": 24})
OUT_FIELDS = [PointField(name=n, offset=OUT_DTYPE.fields[n][1], count=1,
                         datatype=PointField.UINT8 if n == "injected" else PointField.FLOAT32)
              for n in OUT_NAMES]
KINDS = ("multipath", "moving", "clutter")
REFERENCE_HYPOTHESES = 400  # RANSAC samples for the clean scan's ego-velocity
DOPPLER_SIGN = -1.0  # as frontend.cc: -doppler = u^T v


def read_cloud(msg, names):
    """The named fields of a PointCloud2 as float64 columns."""
    fields = {f.name: f for f in msg.fields}
    missing = [n for n in names if n not in fields]
    if missing:
        raise ValueError(f"radar cloud has no fields {missing}")
    dtype = np.dtype({"names": names,
                      "formats": [NUMPY_TYPES[fields[n].datatype] for n in names],
                      "offsets": [fields[n].offset for n in names],
                      "itemsize": msg.point_step})
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.row_step)
    pts = rows[:, :msg.width * msg.point_step].copy().view(dtype).reshape(-1)
    return {n: pts[n].astype(np.float64) for n in names}


def fit_ego_velocity(u, y, threshold, rng):
    """Radar velocity with u v = y, by RANSAC and a least-squares refit. None if too few points."""
    if len(y) < 5:
        return None
    samples = np.array([rng.choice(len(y), 3, replace=False)
                        for _ in range(REFERENCE_HYPOTHESES)])
    A, b = u[samples], y[samples]
    ok = np.abs(np.linalg.det(A)) > 1e-4
    if not ok.any():
        return None
    v = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]  # hypotheses x 3
    support = (np.abs(v @ u.T - y) < threshold).sum(axis=1)
    inliers = np.abs(u @ v[np.argmax(support)] - y) < threshold
    if inliers.sum() < 3:
        return None
    return np.linalg.lstsq(u[inliers], y[inliers], rcond=None)[0]


class OutlierInjector(Node):
    def __init__(self):
        super().__init__("outlier_injector")
        radar_topic = self.declare_parameter("radar_topic", "/continental/points").value
        namespace = self.declare_parameter("output_namespace", "/outlier_ablation").value
        self.levels = [int(v) for v in self.declare_parameter("levels", [0]).value]
        mix = np.array(self.declare_parameter("mix", [0.4, 0.4, 0.2]).value, dtype=float)
        self.mix = mix / mix.sum()  # multipath, moving, clutter
        self.bounce = float(self.declare_parameter("multipath_factor", 2.0).value)
        self.objects = [int(v) for v in self.declare_parameter("moving_objects", [1, 3]).value]
        self.speed = list(self.declare_parameter("object_speed_mps", [2.0, 10.0]).value)
        self.size = float(self.declare_parameter("object_size_m", 3.0).value)
        self.threshold = float(self.declare_parameter("inlier_threshold_mps", 0.15).value)
        self.seed = int(self.declare_parameter("seed", 0).value)
        self.doppler_field = self.declare_parameter("doppler_field", "v").value
        self.rcs_field = self.declare_parameter("rcs_field", "RCS").value
        if not all(0 <= level <= 100 for level in self.levels):
            raise ValueError(f"outlier levels must be in [0, 100]: {self.levels}")
        if len(self.mix) != len(KINDS):
            raise ValueError(f"mix needs one weight per kind: {KINDS}")

        out_qos = QoSProfile(depth=100)
        self.pubs = {level: self.create_publisher(
            PointCloud2, f"{namespace}/p{level:02d}/points", out_qos) for level in self.levels}
        # Best effort, like radom_node, so it also accepts a best-effort player
        self.create_subscription(PointCloud2, radar_topic, self.on_scan, QoSProfile(
            depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(
            f"replacing {self.levels} % of the points of {radar_topic} with outliers "
            f"({', '.join(f'{k} {w:.0%}' for k, w in zip(KINDS, self.mix))}) "
            f"-> {namespace}/pXX/points")

    def on_scan(self, msg):
        cols = read_cloud(msg, ["x", "y", "z", self.doppler_field, self.rcs_field])
        xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
        doppler = cols[self.doppler_field]
        dist = np.linalg.norm(xyz, axis=1)
        usable = np.flatnonzero(np.isfinite(dist) & (dist > 1e-3) & np.isfinite(doppler))
        u = np.zeros_like(xyz)
        u[usable] = xyz[usable] / dist[usable, None]
        stamp = msg.header.stamp
        ego = fit_ego_velocity(u[usable], DOPPLER_SIGN * doppler[usable], self.threshold,
                               np.random.default_rng([self.seed, stamp.sec, stamp.nanosec]))

        for level, pub in self.pubs.items():
            # Same outliers for the same scan in every playback
            rng = np.random.default_rng([self.seed, level, stamp.sec, stamp.nanosec])
            k = min(round(len(xyz) * level / 100), len(usable))
            new_xyz, new_doppler = xyz.copy(), doppler.copy()
            replaced = np.zeros(len(xyz), dtype=bool)
            if k:
                counts = np.floor(self.mix * k).astype(int)
                counts[-1] += k - counts.sum()  # remainder to clutter
                pick = rng.choice(usable, k, replace=False)
                groups = np.split(pick, np.cumsum(counts)[:-1])
                self.multipath(groups[0], dist, u, doppler, new_xyz, new_doppler, rng)
                self.moving(groups[1], dist, u, doppler, new_xyz, new_doppler, rng)
                self.clutter(groups[2], usable, dist, u, doppler, new_xyz, new_doppler, rng)
                replaced[pick] = True

            label = replaced.copy()
            if ego is not None:  # only Doppler outliers count as outliers
                n = np.linalg.norm(new_xyz, axis=1)
                residual = new_xyz @ ego / np.maximum(n, 1e-6) - DOPPLER_SIGN * new_doppler
                label &= np.abs(residual) > self.threshold

            out = np.zeros(len(xyz), dtype=OUT_DTYPE)
            out["x"], out["y"], out["z"] = new_xyz.T
            out["v"], out["RCS"] = new_doppler, cols[self.rcs_field]
            out["injected"] = label

            cloud = PointCloud2()
            cloud.header = msg.header
            cloud.height, cloud.width = 1, len(out)
            cloud.fields = OUT_FIELDS
            cloud.is_bigendian, cloud.is_dense = False, True
            cloud.point_step, cloud.row_step = OUT_DTYPE.itemsize, OUT_DTYPE.itemsize * len(out)
            cloud.data = out.tobytes()
            pub.publish(cloud)

    def multipath(self, idx, dist, u, doppler, xyz, out_doppler, rng):
        """Double-bounce ghosts: along the same ray, `bounce` x range and Doppler."""
        r = self.bounce * dist[idx] + rng.normal(0.0, 0.2, len(idx))
        xyz[idx] = u[idx] * r[:, None]
        out_doppler[idx] = self.bounce * doppler[idx] + rng.normal(0.0, 0.03, len(idx))

    def moving(self, idx, dist, u, doppler, xyz, out_doppler, rng):
        """Objects: points near a seed azimuth pulled to its range, Doppler + object motion."""
        if not len(idx):
            return
        count = min(rng.integers(self.objects[0], self.objects[1] + 1), len(idx))
        azimuth = np.arctan2(u[idx, 1], u[idx, 0])
        seeds = rng.choice(len(idx), count, replace=False)
        # Each point joins the object whose seed is nearest in azimuth
        gap = np.abs(np.angle(np.exp(1j * (azimuth[:, None] - azimuth[seeds][None, :]))))
        owner = np.argmin(gap, axis=1)
        for j, seed in enumerate(seeds):
            members = idx[owner == j]
            heading = rng.uniform(-np.pi, np.pi)
            v_obj = rng.uniform(*self.speed) * np.array([np.cos(heading), np.sin(heading), 0.0])
            r = dist[idx[seed]] + rng.uniform(-self.size / 2, self.size / 2, len(members))
            xyz[members] = u[members] * np.maximum(r, 0.5)[:, None]
            # A target moving with v_obj: -doppler = u^T (v_ego - v_obj), so it adds u^T v_obj
            out_doppler[members] = doppler[members] + u[members] @ v_obj

    def clutter(self, idx, usable, dist, u, doppler, xyz, out_doppler, rng):
        """Unstructured: direction, range and Doppler drawn across the scan's spread."""
        n = len(idx)
        if not n:
            return
        az = np.arctan2(u[usable, 1], u[usable, 0])
        el = np.arcsin(np.clip(u[usable, 2], -1, 1))
        a = rng.uniform(az.min(), az.max(), n)
        e = rng.uniform(el.min(), el.max(), n)
        r = rng.uniform(dist[usable].min(), dist[usable].max(), n)
        xyz[idx] = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1) * r[:, None]
        lo, hi = doppler[usable].min() - 2.0, doppler[usable].max() + 2.0
        out_doppler[idx] = rng.uniform(lo, hi, n)


def main():
    rclpy.init()
    node = OutlierInjector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
