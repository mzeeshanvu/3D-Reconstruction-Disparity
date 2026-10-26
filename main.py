#!/usr/bin/env python
"""
3D reconstruction from a stereo disparity map -- entry point.

Takes a calibration set and one rectified stereo pair, and produces a dense
coloured point cloud. Every stage writes a figure to output/, so the run can
be read as well as executed.

Usage
-----
    python main.py                            # defaults, on data/
    python main.py --method bm                # block matching instead of SGBM
    python main.py --baseline 0.12            # metric output, if you measured it
    python main.py --compare                  # BM and SGBM side by side
    python main.py --help                     # every option

Each stage below is a function; `main()` calls them in order and records what
each produced.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from stereo import calibration, depth as depth_mod, disparity as disp_mod
from stereo import export, rectify, visualization as viz

REPO_ROOT = Path(__file__).resolve().parent


class Tee:
    """Mirror stdout to a log file so the run transcript is preserved."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def banner(text):
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)


def parse_pattern(text):
    """`--pattern 9x6` into (9, 6)."""
    try:
        cols, rows = text.lower().split("x")
        return int(cols), int(rows)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected COLSxROWS, e.g. 9x6, got {text!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reconstruct a dense 3D point cloud from a stereo pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = p.add_argument_group("input")
    data.add_argument("--calibration-dir", default="data/calibration",
                      help="folder of chessboard images")
    data.add_argument("--calibration-glob", default="cal*.png",
                      help="which files in that folder to use")
    data.add_argument("--pattern", type=parse_pattern, default="9x6",
                      help="chessboard INTERIOR corners, as COLSxROWS")
    data.add_argument("--left", default="data/stereo/left1.png")
    data.add_argument("--right", default="data/stereo/right1.png")

    match = p.add_argument_group("matching")
    match.add_argument("--method", default="sgbm", choices=["sgbm", "bm"],
                       help="semi-global or plain block matching")
    match.add_argument("--num-disparities", type=int, default=None,
                       help="search width, a multiple of 16; measured from "
                            "the image pair when omitted")
    match.add_argument("--block-size", type=int, default=5,
                       help="correlation window, odd")
    match.add_argument("--no-fill", action="store_true",
                       help="do not interpolate across narrow holes")
    match.add_argument("--compare", action="store_true",
                       help="also run the other matcher and write a "
                            "comparison figure")

    geom = p.add_argument_group("geometry")
    geom.add_argument("--baseline", type=float, default=None,
                      help="distance between the two camera centres, in "
                           "whatever units you measured. Without it, depth "
                           "is reported in units of the baseline")
    geom.add_argument("--max-depth", type=float, default=None,
                      help="discard points beyond this; a percentile of the "
                           "data is used when omitted")
    geom.add_argument("--skip-rectify-check", action="store_true",
                      help="do not verify that the pair is rectified")

    p.add_argument("--out", default="output",
                   help="where results are written")

    args = p.parse_args(argv)

    # Validated here rather than left to OpenCV, which reports these as a
    # bare exception from inside the matcher.
    if args.num_disparities is not None and (
            args.num_disparities <= 0 or args.num_disparities % 16):
        p.error(f"--num-disparities must be a positive multiple of 16, "
                f"got {args.num_disparities}")
    if args.block_size % 2 == 0 or args.block_size < 1:
        p.error(f"--block-size must be a positive odd number, "
                f"got {args.block_size}")
    return args


# ==========================================================================
# Stages
# ==========================================================================

def stage_calibrate(args, out_dir):
    """Stage 1 -- recover the intrinsics from the chessboard images."""
    banner("STAGE 1 of 5 -- CAMERA CALIBRATION")
    calib = calibration.calibrate(
        REPO_ROOT / args.calibration_dir, pattern=args.pattern,
        glob=args.calibration_glob)
    print(f"  {calib.summary()}")
    print(f"  images {', '.join(calib.used)}")

    sample = REPO_ROOT / args.calibration_dir / calib.used[0]
    viz.plot_calibration(calib, sample, args.pattern,
                         out_dir / "01_calibration.png")
    print("  wrote 01_calibration.png")
    return calib, {
        "n_images": calib.n_images, "rms_reprojection_px": calib.rms,
        "fx": calib.fx, "fy": calib.fy, "cx": calib.cx, "cy": calib.cy,
        "distortion": [float(v) for v in calib.dist],
    }


def stage_prepare(args, calib, out_dir):
    """Stage 2 -- load the pair, undistort it, and check it is rectified."""
    banner("STAGE 2 of 5 -- STEREO PAIR AND RECTIFICATION CHECK")

    left = cv2.imread(str(REPO_ROOT / args.left))
    right = cv2.imread(str(REPO_ROOT / args.right))
    for img, path in ((left, args.left), (right, args.right)):
        if img is None:
            raise SystemExit(f"\nCould not read {REPO_ROOT / path}")
    if left.shape != right.shape:
        raise SystemExit(
            f"\nThe two images differ in size: {left.shape[:2][::-1]} vs "
            f"{right.shape[:2][::-1]}. A stereo pair must match.")
    print(f"  {left.shape[1]}x{left.shape[0]}  "
          f"{Path(args.left).name} / {Path(args.right).name}")

    # Undistortion comes before matching, not after: block matching assumes
    # straight epipolar lines, and lens distortion curves them.
    if calib.image_size != (left.shape[1], left.shape[0]):
        print(f"  note: calibrated at {calib.image_size}, pair is "
              f"{(left.shape[1], left.shape[0])}; skipping undistortion")
        stats_undistort = False
    else:
        left = calibration.undistort(left, calib)
        right = calibration.undistort(right, calib)
        stats_undistort = True
        print("  undistorted both views with the calibrated coefficients")

    stats = {}
    if not args.skip_rectify_check:
        stats = rectify.check_rectified(left, right)
        print(rectify.describe(stats))
        viz.plot_rectification(left, right, stats,
                               out_dir / "02_rectification.png")
        print("  wrote 02_rectification.png")

    stats["undistorted"] = stats_undistort
    return left, right, stats


def stage_disparity(args, left, right, rect_stats, out_dir):
    """Stage 3 -- the disparity map."""
    banner("STAGE 3 of 5 -- DISPARITY")

    num = args.num_disparities
    if num is None:
        num = rectify.suggest_num_disparities(rect_stats) if rect_stats else 64
        print(f"  search width measured from the pair: {num} px")
    else:
        print(f"  search width: {num} px")

    started = time.time()
    d = disp_mod.compute(left, right, method=args.method,
                         num_disparities=num, block_size=args.block_size)
    raw_valid = disp_mod.valid_fraction(d)
    print(f"  {args.method.upper()}, block {args.block_size}: "
          f"{raw_valid:.1%} of pixels matched  ({time.time() - started:.2f}s)")

    if not args.no_fill:
        d = disp_mod.fill_small_holes(d)
        print(f"  after filling narrow holes: "
              f"{disp_mod.valid_fraction(d):.1%}")

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        raise SystemExit(
            "\nNo pixel matched at all. The pair may not be rectified, the "
            "left and right images may be swapped, or --num-disparities may "
            "be too small for the scene.")
    print(f"  disparity {finite.min():.2f}..{finite.max():.2f} px "
          f"(median {np.median(finite):.2f})")

    viz.plot_disparity(left, d, out_dir / "03_disparity.png",
                       method=args.method)
    viz.save_disparity_image(d, out_dir / "03_disparity_map.png")
    print("  wrote 03_disparity.png, 03_disparity_map.png")

    if args.compare:
        other = "bm" if args.method == "sgbm" else "sgbm"
        alt = disp_mod.compute(left, right, method=other,
                               num_disparities=num,
                               block_size=args.block_size)
        viz.plot_method_comparison(
            left, {args.method: d, other: alt},
            out_dir / "03_method_comparison.png")
        print(f"  wrote 03_method_comparison.png "
              f"({other.upper()}: {disp_mod.valid_fraction(alt):.1%} valid)")

    return d, {
        "method": args.method, "num_disparities": num,
        "block_size": args.block_size,
        "valid_fraction_raw": raw_valid,
        "valid_fraction_final": disp_mod.valid_fraction(d),
        "disparity_min_px": float(finite.min()),
        "disparity_max_px": float(finite.max()),
        "disparity_median_px": float(np.median(finite)),
    }


def stage_depth(args, d, calib, out_dir):
    """Stage 4 -- disparity to depth."""
    banner("STAGE 4 of 5 -- DEPTH")

    baseline = args.baseline if args.baseline is not None else 1.0
    known = args.baseline is not None
    unit = "metric units" if known else "baseline units"

    Z = depth_mod.depth_from_disparity(d, calib.fx, baseline)
    pct = depth_mod.depth_percentiles(Z)
    print(f"  Z = fx*B/d  with fx = {calib.fx:.2f} px, B = "
          f"{baseline:g} ({'measured' if known else 'unknown, taken as 1'})")
    print(f"  depth {pct['p1']:.2f}..{pct['p99']:.2f} {unit} "
          f"(median {pct['p50']:.2f})")
    if not known:
        print("  the baseline was not supplied, so these are relative "
              "depths: the shape of the scene is correct, its size is not")

    viz.plot_depth(Z, out_dir / "04_depth.png", known, unit)
    print("  wrote 04_depth.png")
    return Z, {"baseline": baseline, "baseline_known": known,
               "unit": unit, "percentiles": pct}


def stage_cloud(args, Z, left, calib, unit, out_dir):
    """Stage 5 -- back-projection to a coloured point cloud."""
    banner("STAGE 5 of 5 -- POINT CLOUD")

    max_depth = args.max_depth
    if max_depth is None:
        max_depth = depth_mod.suggest_max_depth(Z)
        print(f"  far cut-off from the 97th percentile: {max_depth:.2f} {unit}")

    points, colors = depth_mod.back_project(Z, calib.K, colors=left,
                                            max_depth=max_depth)
    print(f"  {len(points):,} points")
    if len(points) == 0:
        raise SystemExit("\nEvery point was rejected; try a larger "
                         "--max-depth.")

    ply, n = export.write_ply(out_dir / "point_cloud.ply", points, colors)
    export.write_xyz(out_dir / "point_cloud.xyz", points, colors)
    print(f"  wrote point_cloud.ply ({n:,} points), point_cloud.xyz")

    viz.plot_point_cloud(points, colors, out_dir / "05_point_cloud.png", unit)
    print("  wrote 05_point_cloud.png")
    return {"n_points": int(n), "max_depth": float(max_depth),
            "ply": str(ply)}


# ==========================================================================

def main(argv=None):
    args = parse_args(argv)
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tee = Tee(out_dir / "run_log.txt")
    sys.stdout = tee
    started = time.time()

    try:
        banner("3D RECONSTRUCTION VIA STEREO DISPARITY")
        print(f"  output: {out_dir}")

        calib, s_calib = stage_calibrate(args, out_dir)
        left, right, s_rect = stage_prepare(args, calib, out_dir)
        d, s_disp = stage_disparity(args, left, right, s_rect, out_dir)
        Z, s_depth = stage_depth(args, d, calib, out_dir)
        s_cloud = stage_cloud(args, Z, left, calib, s_depth["unit"], out_dir)

        stats = {
            "runtime_seconds": round(time.time() - started, 2),
            "stage_1_calibration": s_calib,
            "stage_2_rectification": s_rect,
            "stage_3_disparity": s_disp,
            "stage_4_depth": s_depth,
            "stage_5_cloud": s_cloud,
        }
        export.write_stats(stats, out_dir / "stats.json")

        banner("DONE")
        print(f"  calibration RMS   {calib.rms:.4f} px")
        print(f"  pixels matched    {s_disp['valid_fraction_final']:.1%}")
        print(f"  points            {s_cloud['n_points']:,}")
        print(f"  runtime           {stats['runtime_seconds']}s")
        print(f"\n  point cloud: {out_dir / 'point_cloud.ply'}")
        if not s_depth["baseline_known"]:
            print("\n  Depth is in units of the baseline. Measure the "
                  "distance between\n  the two camera centres and pass it "
                  "as --baseline to make it metric.")
        return 0

    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    sys.exit(main())
