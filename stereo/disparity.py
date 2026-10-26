"""
Disparity: how far each pixel moved between the two views.

A rectified pair reduces correspondence to a one-dimensional search. For a
pixel at (u, v) in the left image, the matching pixel in the right image is
at (u - d, v) for some d >= 0, and d is the disparity. Nearer surfaces move
further, so d is the quantity depth is read from.

Two matchers are offered.

``StereoBM`` is the classic block matcher: slide a window along the row and
take the position with the lowest sum of absolute differences. It is fast and
it is what the original study used, but it decides each pixel independently,
so it falls apart on low-texture regions -- a blank wall or a tabletop offers
nothing to match and the result is noise.

``StereoSGBM`` adds a smoothness penalty aggregated along several directions,
which propagates confident matches into the places block matching cannot
decide for itself. It is the better default, especially here: a narrow
disparity range needs the sub-pixel interpolation SGBM does and BM does not.

Fixed-point output
------------------
Both matchers return **16 x disparity as a 16-bit signed integer**, not
disparity. Dividing that out is not optional: a depth computed from the raw
value is 16 times too small, and nothing in the result looks wrong enough to
notice. :func:`compute` returns float pixels, already divided.

Invalid pixels -- occluded, or outside the search range -- come back as a
negative sentinel, and are returned here as NaN so they cannot be mistaken
for a real measurement of zero.
"""

import cv2
import numpy as np


def _as_gray(image):
    return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image)


def make_matcher(method="sgbm", num_disparities=64, block_size=5,
                 min_disparity=0):
    """
    Build a matcher.

    :param num_disparities: width of the search, which OpenCV requires to be
                            a positive multiple of 16
    :param block_size: the correlation window; odd. Small windows resolve
                       detail but are noisier, large ones smooth over depth
                       discontinuities and bleed edges
    """
    if num_disparities % 16 or num_disparities <= 0:
        raise ValueError(
            f"num_disparities must be a positive multiple of 16, "
            f"got {num_disparities}")
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")

    if method == "bm":
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)
        matcher.setMinDisparity(min_disparity)
        return matcher

    if method == "sgbm":
        # P1 and P2 penalise disparity changing by one and by more than one
        # between neighbouring pixels. The 8 and 32 multipliers are the
        # values the OpenCV documentation recommends; they scale with the
        # window area because the penalty is compared against a cost summed
        # over that window.
        channels = 1
        return cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * channels * block_size ** 2,
            P2=32 * channels * block_size ** 2,
            disp12MaxDiff=1,            # left-right consistency, in pixels
            uniquenessRatio=10,         # reject a match the runner-up nearly ties
            speckleWindowSize=100,      # remove small inconsistent blobs
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    raise ValueError(f"unknown method {method!r}; expected 'bm' or 'sgbm'")


def compute(left, right, method="sgbm", num_disparities=64, block_size=5,
            min_disparity=0):
    """
    The disparity map, in pixels.

    :returns: (h, w) float32, NaN where no valid match was found
    """
    matcher = make_matcher(method, num_disparities, block_size, min_disparity)
    raw = matcher.compute(_as_gray(left), _as_gray(right))

    # The 16x fixed-point scale OpenCV returns. Skipping this is the single
    # most common error in stereo code and it is invisible downstream: every
    # depth simply comes out 16 times too small.
    disparity = raw.astype(np.float32) / 16.0

    # Anything at or below the lowest searched disparity was never matched.
    disparity[raw <= (min_disparity * 16)] = np.nan
    return disparity


def valid_fraction(disparity):
    """Share of pixels that received a real match."""
    return float(np.isfinite(disparity).mean())


def fill_small_holes(disparity, max_hole=5):
    """
    Interpolate across narrow gaps, leaving large ones alone.

    Small holes are usually a textureless speck or a single failed match, and
    filling them tidies the surface. Large holes are genuine occlusions --
    surface visible to one camera only -- where there is no measurement to
    recover and inventing one would be fabricating geometry.

    :param max_hole: widest run of invalid pixels, in pixels, to fill
    """
    out = disparity.copy()
    invalid = ~np.isfinite(out)
    if not invalid.any():
        return out

    mask = invalid.astype(np.uint8)
    # A hole narrower than max_hole survives neither erosion nor the opening
    # below, so what remains marks the holes too big to fill honestly.
    kernel = np.ones((1, max_hole * 2 + 1), np.uint8)
    large = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    fillable = (mask == 1) & (large == 0)
    if not fillable.any():
        return out

    filled = cv2.inpaint(
        np.nan_to_num(out, nan=0.0).astype(np.float32),
        fillable.astype(np.uint8), 3, cv2.INPAINT_NS)
    out[fillable] = filled[fillable]
    return out


def for_display(disparity):
    """
    An 8-bit image of the disparity, for looking at only.

    Kept separate from :func:`compute` on purpose. Normalising to 0..255 is
    exactly what the original study did *before* computing depth, which
    destroys the metric disparity and silently invalidates every distance
    derived from it. Normalise for the eye; never for the arithmetic.
    """
    finite = np.isfinite(disparity)
    out = np.zeros(disparity.shape, np.uint8)
    if not finite.any():
        return out
    lo, hi = np.percentile(disparity[finite], [1, 99])
    if hi <= lo:
        return out
    scaled = np.clip((disparity - lo) / (hi - lo), 0, 1)
    out[finite] = (scaled[finite] * 255).astype(np.uint8)
    return out
