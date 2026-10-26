"""
Camera calibration from chessboard images (Zhang's method).

The intrinsics are what turn a disparity into a distance. Without them a
disparity map is only a picture of which pixels moved further -- `fx` is the
term that converts the pixel shift into a depth, and the principal point is
what the back-projection measures its rays from.

A planar chessboard is used because its geometry is known exactly up to one
scale: the corners are coplanar and evenly spaced, so each view gives a
homography, and several views in different orientations over-determine the
intrinsics. See Zhang, *A Flexible New Technique for Camera Calibration*,
PAMI 2000.

The square size is deliberately left in arbitrary units. It scales the
translation of each board, which this module discards, but never the camera
matrix, which it returns -- so a calibration done in "squares" gives exactly
the same `K` in pixels as one done in millimetres.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Inner corners, not squares: a board of 10x7 squares has 9x6 interior
# corners, and it is the corners that are detected.
DEFAULT_PATTERN = (9, 6)

# Sub-pixel refinement stops at whichever comes first.
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@dataclass
class Calibration:
    """The result of calibrating one camera."""

    K: np.ndarray            # (3, 3) camera matrix, in pixels
    dist: np.ndarray         # (5,) radial and tangential distortion
    rms: float               # RMS reprojection error, in pixels
    image_size: tuple        # (width, height) the calibration was done at
    n_images: int            # how many views contributed
    used: list               # the filenames that contributed

    @property
    def fx(self):
        return float(self.K[0, 0])

    @property
    def fy(self):
        return float(self.K[1, 1])

    @property
    def cx(self):
        return float(self.K[0, 2])

    @property
    def cy(self):
        return float(self.K[1, 2])

    def summary(self):
        return (f"fx {self.fx:.2f}  fy {self.fy:.2f}  "
                f"cx {self.cx:.2f}  cy {self.cy:.2f}  "
                f"RMS {self.rms:.4f} px  ({self.n_images} views)")


def find_corners(path, pattern=DEFAULT_PATTERN):
    """
    Locate the chessboard's interior corners in one image.

    Refined to sub-pixel accuracy, because the whole calibration is only as
    good as these coordinates -- an integer-pixel corner caps the achievable
    reprojection error at roughly half a pixel no matter how many views are
    supplied.

    :returns: (n, 1, 2) corners, or None when no board is visible
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)

    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not found:
        return None, gray
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA), gray


def object_points(pattern=DEFAULT_PATTERN, square_size=1.0):
    """
    The board's corners in its own coordinate frame, z = 0.

    One grid, reused for every view: the board is rigid, so what changes
    between views is its pose, which calibration solves for.
    """
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid * square_size


def calibrate(image_dir, pattern=DEFAULT_PATTERN, glob="cal*.png",
              square_size=1.0):
    """
    Calibrate one camera from a folder of chessboard images.

    :param image_dir: folder holding the calibration photographs
    :param pattern: interior corners as (columns, rows)
    :param square_size: side of one square; affects nothing this returns
    :returns: :class:`Calibration`
    :raises SystemExit: when too few boards are found to solve
    """
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob(glob))
    if not files:
        raise SystemExit(f"\nNo images matching '{glob}' in {image_dir}")

    grid = object_points(pattern, square_size)
    obj_points, img_points, used = [], [], []
    size = None

    for path in files:
        corners, gray = find_corners(path, pattern)
        if corners is None:
            continue
        obj_points.append(grid)
        img_points.append(corners)
        used.append(path.name)
        size = gray.shape[::-1]          # (width, height)

    # Each view contributes two constraints on the intrinsics, and there are
    # four unknowns in K, so three views is the theoretical floor. Fewer than
    # that and the distortion coefficients in particular are unconstrained.
    if len(obj_points) < 3:
        raise SystemExit(
            f"\nFound a {pattern[0]}x{pattern[1]} chessboard in only "
            f"{len(obj_points)} of {len(files)} images in {image_dir}.\n"
            f"Three is the minimum; if the board has a different number of "
            f"interior corners, pass --pattern COLSxROWS.")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, size, None, None)

    return Calibration(K=K, dist=dist.ravel(), rms=float(rms),
                       image_size=size, n_images=len(obj_points), used=used)


def undistort(image, calib):
    """
    Remove lens distortion, keeping the original camera matrix.

    `cv2.getOptimalNewCameraMatrix` is deliberately not used: it returns a
    *different* matrix, and every depth computed downstream would then have
    to use that one instead. Keeping `K` fixed means the fx in `Z = fx*B/d`
    is the fx that was calibrated.
    """
    return cv2.undistort(image, calib.K, calib.dist, None, calib.K)
