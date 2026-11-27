#!/usr/bin/env python3
"""Show a HeRCULES left camera image next to the radar scan taken at the same time.

The radar scan is drawn as a 2D range image: each point sits at its (azimuth,
elevation) as seen from the sensor, its colour is its range, and where points
overlap the nearer one is kept. Left is left in both panels.

    python3 src/radom/scripts/radar_to_image.py mountain             # middle of the drive
    python3 src/radom/scripts/radar_to_image.py mountain --time 120  # 120 s in

Radar binary layout (packed, little-endian, 29 bytes per point, as read by the
official file player): x y z v r [f32], RCS [i8], azimuth elevation [f32, rad].
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("v", "<f4"),
                  ("r", "<f4"), ("RCS", "i1"),
                  ("azimuth", "<f4"), ("elevation", "<f4")])


def find_sequence(dataset, name):
    """'mountain' -> dataset/hercules_mountain_01 (must match exactly one folder)."""
    matches = sorted(d for d in dataset.iterdir()
                     if d.is_dir() and name in d.name and (d / "Radar" / "Continental").is_dir())
    if len(matches) != 1:
        found = ", ".join(d.name for d in matches) or "none"
        raise SystemExit(f"'{name}' must match one sequence in {dataset} (matched: {found})")
    return matches[0]


def stamps(folder, suffix):
    """Files are named <nanosecond stamp><suffix>."""
    files = sorted(folder.glob(f"*{suffix}"))
    if not files:
        raise SystemExit(f"no {suffix} files in {folder}")
    return np.array([int(f.stem) for f in files]), files


def range_image(points, az_fov, el_fov, res, radius):
    """Image rows are elevation (up is up), columns azimuth (left is left)."""
    width = int(round(2 * az_fov / res))
    height = int(round(2 * el_fov / res))
    image = np.full((height, width), np.nan)

    az = np.degrees(points["azimuth"])
    el = np.degrees(points["elevation"])
    col = np.round((az_fov - az) / res).astype(int)  # +azimuth is to the left
    row = np.round((el_fov - el) / res).astype(int)

    # Far points first, so nearer ones overwrite them
    for i in np.argsort(points["r"])[::-1]:
        r0, r1 = max(row[i] - radius, 0), min(row[i] + radius + 1, height)
        c0, c1 = max(col[i] - radius, 0), min(col[i] + radius + 1, width)
        if r0 < r1 and c0 < c1:
            image[r0:r1, c0:c1] = points["r"][i]
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sequence", help="part of a sequence folder name, e.g. mountain")
    parser.add_argument("--dataset", type=Path, default=REPO / "dataset")
    parser.add_argument("--time", type=float, help="seconds from the first scan; default: middle")
    parser.add_argument("--out", type=Path, help="default: docs/images/<sequence>_<stamp>.png")
    parser.add_argument("--az-fov", type=float, default=60.0, help="half FOV [deg]")
    parser.add_argument("--el-fov", type=float, default=25.0, help="half FOV [deg]")
    parser.add_argument("--res", type=float, default=0.25, help="[deg / pixel]")
    parser.add_argument("--radius", type=int, default=2, help="point size [pixels]")
    parser.add_argument("--max-range", type=float,
                        help="colour scale limit [m]; default: 98th percentile")
    parser.add_argument("--show", action="store_true", help="also open a window")
    args = parser.parse_args()

    sequence = find_sequence(args.dataset, args.sequence)
    radar_t, radar_files = stamps(sequence / "Radar" / "Continental", ".bin")
    camera_t, camera_files = stamps(sequence / "Image" / "stereo_left", ".png")

    # Radar scan nearest the requested time, then the camera frame nearest that scan
    target = (radar_t[0] + radar_t[-1]) // 2 if args.time is None \
        else radar_t[0] + int(args.time * 1e9)
    i = int(np.argmin(np.abs(radar_t - target)))
    j = int(np.argmin(np.abs(camera_t - radar_t[i])))
    gap_ms = (camera_t[j] - radar_t[i]) / 1e6
    t_s = (radar_t[i] - radar_t[0]) / 1e9

    points = np.fromfile(radar_files[i], dtype=POINT)
    if points.size == 0:
        raise SystemExit(f"{radar_files[i]}: no points")
    image = range_image(points, args.az_fov, args.el_fov, args.res, args.radius)
    camera = plt.imread(camera_files[j])

    # Equal panel heights: widths follow each panel's aspect ratio
    radar_aspect = args.az_fov / args.el_fov
    camera_aspect = camera.shape[1] / camera.shape[0]
    fig, (ax_cam, ax_radar) = plt.subplots(
        1, 2, figsize=(4.5 * (camera_aspect + radar_aspect) + 1.5, 5),
        gridspec_kw={"width_ratios": [camera_aspect, radar_aspect]}, layout="constrained")

    ax_cam.imshow(camera)
    ax_cam.set_axis_off()
    ax_cam.set_title(f"left camera ({gap_ms:+.0f} ms)")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")  # no return
    shown = ax_radar.imshow(
        image, cmap=cmap, vmin=0, vmax=args.max_range or np.percentile(points["r"], 98),
        extent=[args.az_fov, -args.az_fov, -args.el_fov, args.el_fov],
        aspect="auto", interpolation="nearest")
    ax_radar.set_xlabel("azimuth [deg]")
    ax_radar.set_ylabel("elevation [deg]")
    ax_radar.set_title(f"radar: {points.size} points")
    fig.colorbar(shown, ax=ax_radar, label="range [m]", fraction=0.04, pad=0.02)
    fig.suptitle(f"{sequence.name}, t = {t_s:.1f} s")

    out = args.out or REPO / "docs" / "images" / f"{sequence.name}_{radar_t[i]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
