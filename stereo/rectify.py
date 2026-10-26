"""
Checking -- and if necessary enforcing -- that the stereo pair is rectified.

Block matching searches along image rows. It assumes the two views are
*rectified*: that corresponding points sit on the same row in both images, so
a correspondence is a pure horizontal shift. If that assumption is wrong the
matcher is looking in the wrong place, and the disparity map it returns is
confidently incorrect rather than obviously broken. That failure is silent,
which is why this module measures the assumption instead of trusting it.

The measurement is direct: match features between the two images, keep the
geometrically consistent ones, and look at how far apart the matches are
*vertically*. On a rectified pair that number is zero to within noise.

Full rectification needs the extrinsics between the two cameras -- the
rotation and translation of one relative to the other -- which come from
stereo calibration, not from calibrating a single camera on a chessboard.
When those are not available, verification is still possible and is what this
module provides; see :func:`check_rectified`.
"""

import cv2
import numpy as np

#: Mean absolute vertical disparity, in pixels, below which a pair is treated
#: as rectified. Sub-pixel is the standard a rectified pair should meet; a
#: value of several pixels means rows do not correspond and block matching
#: will fail.
TOLERANCE = 1.0


def matched_points(left, right, ratio=0.7):
    """
    Corresponding points between two images, via SIFT and the ratio test.

    :param ratio: Lowe's ratio; lower keeps fewer but surer matches, which is
                  what this is for -- the question is whether good matches lie
                  on the same row, so ambiguous ones only add noise
    :returns: (points_left (n, 2), points_right (n, 2))
    """
    if left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    if right.ndim == 3:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(left, None)
    kp2, desc2 = sift.detectAndCompute(right, None)
    if desc1 is None or desc2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))

    pairs = cv2.BFMatcher().knnMatch(desc1, desc2, k=2)
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if not good:
        return np.zeros((0, 2)), np.zeros((0, 2))

    p1 = np.float32([kp1[m.queryIdx].pt for m in good])
    p2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return p1, p2


def check_rectified(left, right, tolerance=TOLERANCE):
    """
    Measure how rectified a pair actually is.

    Repetitive texture -- a bookshelf, a brick wall, a keyboard -- produces
    matches that pass the ratio test but join the wrong instance of a
    repeated pattern. Those land on the wrong row and would dominate the
    statistic, so the fundamental matrix is fitted with RANSAC first and only
    its inliers are measured.

    :returns: a dict with the vertical-disparity statistics, the horizontal
              disparity range the matcher will need to cover, and a
              ``rectified`` flag
    """
    p1, p2 = matched_points(left, right)
    if len(p1) < 8:
        return {"n_matches": len(p1), "rectified": False,
                "reason": "too few matches to judge"}

    _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0)
    if mask is None:
        return {"n_matches": len(p1), "rectified": False,
                "reason": "no consistent two-view geometry"}

    inliers = mask.ravel() == 1
    a, b = p1[inliers], p2[inliers]
    dy = b[:, 1] - a[:, 1]
    dx = a[:, 0] - b[:, 0]          # positive for a conventional left/right pair

    mean_abs_dy = float(np.abs(dy).mean())
    return {
        "n_matches": int(len(p1)),
        "n_inliers": int(inliers.sum()),
        "mean_abs_dy": mean_abs_dy,
        "std_dy": float(dy.std()),
        "max_abs_dy": float(np.abs(dy).max()),
        "disparity_min": float(np.percentile(dx, 1)),
        "disparity_max": float(np.percentile(dx, 99)),
        "disparity_median": float(np.median(dx)),
        "negative_fraction": float((dx < 0).mean()),
        "rectified": mean_abs_dy < tolerance,
    }


def describe(stats):
    """A human-readable verdict, for the console and the run log."""
    if "reason" in stats:
        return f"cannot verify rectification: {stats['reason']}"

    verdict = "rectified" if stats["rectified"] else "NOT rectified"
    lines = [
        f"  {verdict}: mean |dy| = {stats['mean_abs_dy']:.2f} px "
        f"(std {stats['std_dy']:.2f}, max {stats['max_abs_dy']:.2f}) "
        f"over {stats['n_inliers']} inliers",
        f"  horizontal disparity spans "
        f"{stats['disparity_min']:.1f}..{stats['disparity_max']:.1f} px "
        f"(median {stats['disparity_median']:.1f})",
    ]
    if not stats["rectified"]:
        lines.append(
            "  block matching searches along rows, so a pair this far from "
            "rectified will give a confidently wrong disparity map")
    if stats.get("negative_fraction", 0) > 0.2:
        lines.append(
            f"  {stats['negative_fraction']:.0%} of matches shift the wrong "
            f"way -- the left and right images may be swapped")
    return "\n".join(lines)


def suggest_num_disparities(stats, margin=1.5, default=64):
    """
    A `numDisparities` that covers the measured range.

    OpenCV requires a multiple of 16, and the cost of the search grows with
    it, so the useful value is the smallest multiple that still contains the
    real disparities with room to spare. Guessing high is not free: a wider
    search admits more false matches in textureless regions.

    :param margin: headroom over the measured maximum, since sparse feature
                   matches will miss the very nearest surfaces
    :param default: returned when `stats` carries no measurement -- the
                    check was skipped, or it failed. Guessing the floor of
                    16 there would quietly clip every near surface, which is
                    the worse failure: too wide only costs time.
    """
    hi = stats.get("disparity_max")
    if hi is None:
        return int(default)
    return int(max(16, int(np.ceil(hi * margin / 16.0)) * 16))


def rectify_pair(left, right, K, dist, R, T):
    """
    Rectify a pair using known stereo extrinsics.

    Provided for completeness, and used only when `R` and `T` between the two
    cameras are available -- that is, after a *stereo* calibration, where the
    chessboard was photographed by both cameras simultaneously. Calibrating a
    single camera on a chessboard, as :mod:`stereo.calibration` does, cannot
    produce them.

    :param R: (3, 3) rotation of the right camera relative to the left
    :param T: (3,) translation of the right camera relative to the left
    :returns: (rectified_left, rectified_right, Q) where Q reprojects
              disparity to 3D
    """
    size = (left.shape[1], left.shape[0])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist, K, dist, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, size,
                                               cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, size,
                                               cv2.CV_32FC1)
    return (cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR),
            cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR),
            Q)
