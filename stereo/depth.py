"""
Depth and 3D points from disparity.

For a rectified pair the relation is exact and follows from similar
triangles:

    Z = fx * B / d

where `fx` is the focal length in pixels, `B` the baseline -- the distance
between the two camera centres -- and `d` the disparity in pixels. Having Z,
the other two coordinates come from inverting the pinhole projection:

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

On the baseline
---------------
`B` is the one quantity in that equation which cannot be recovered from the
images. It is a physical distance between two cameras, and it enters as a
pure multiplier: every Z, X and Y scales linearly with it.

Calibrating a single camera on a chessboard -- which is what a folder of
chessboard photographs supports -- yields `K` and the distortion, and no
baseline at all. Recovering one needs either a *stereo* calibration, with the
board photographed by both cameras at the same moment, or a measurement with
a ruler.

So this module does not invent a value. With no baseline given it reports
depth in **units of the baseline**, Z/B = fx/d, which is the complete and
honest result: the shape of the scene is fully determined, its size is not.
Supply a measured `--baseline` and the same numbers become metric. This
mirrors Structure from Motion, where translation is likewise recovered only
up to scale.
"""

import numpy as np


def depth_from_disparity(disparity, fx, baseline=1.0):
    """
    ``Z = fx * B / d``, elementwise.

    :param disparity: (h, w) in pixels, NaN where invalid
    :param baseline: distance between the camera centres; left at 1.0 the
                     result is in units of the baseline
    :returns: (h, w) depth, NaN where disparity was invalid or non-positive
    """
    d = np.asarray(disparity, np.float64)

    # Zero or negative disparity is a point at or beyond infinity. It is not
    # a depth of zero, and dividing by it would produce an infinity that
    # propagates into every downstream statistic.
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(d > 0, fx * baseline / d, np.nan)
    return Z


def back_project(depth, K, colors=None, max_depth=None):
    """
    Lift a depth map into a 3D point cloud.

    Vectorised over the whole image rather than looped per pixel: the loop in
    the original study ran for minutes on a 400x300 image and produced the
    same answer this produces in milliseconds.

    :param depth: (h, w), NaN where there is no measurement
    :param K: (3, 3) camera matrix
    :param colors: (h, w, 3) BGR image to colour the points with
    :param max_depth: discard points beyond this, in the same units as
                      `depth`. Disparity error grows as Z^2, so the far tail
                      is both the least accurate part of the cloud and the
                      part that stretches its bounding box most.
    :returns: (points (n, 3), colors (n, 3) RGB in 0..1 or None)
    """
    depth = np.asarray(depth, np.float64)
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    if not valid.any():
        return np.zeros((0, 3)), (np.zeros((0, 3)) if colors is not None else None)

    v, u = np.nonzero(valid)
    Z = depth[v, u]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points = np.stack([X, Y, Z], axis=1)

    rgb = None
    if colors is not None:
        # OpenCV hands over BGR; PLY and every viewer expect RGB.
        rgb = colors[v, u][:, ::-1].astype(np.float64) / 255.0
    return points, rgb


def depth_percentiles(depth, qs=(1, 25, 50, 75, 99)):
    """Percentiles over the valid pixels, for the run log."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {}
    return {f"p{q}": float(np.percentile(finite, q)) for q in qs}


def suggest_max_depth(depth, percentile=97.0):
    """
    A cut-off that keeps the scene and drops the far tail.

    Because Z goes as 1/d, the handful of pixels with the smallest disparities
    -- often the least reliable matches in the image -- land arbitrarily far
    away and dominate the bounding box. A high percentile keeps the structure
    and removes them without a hand-tuned constant.
    """
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))
