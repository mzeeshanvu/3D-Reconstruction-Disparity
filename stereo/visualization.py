"""
The figures. Every stage writes one, so the pipeline can be read as well as
run.

Matplotlib is forced onto the Agg backend: the pipeline writes files and
never opens a window, and on a headless machine the interactive backends
fail at import.
"""

import matplotlib

matplotlib.use("Agg")

import cv2                                            # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

from . import disparity as disp_mod                   # noqa: E402


def _rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_calibration(calib, sample_path, pattern, path):
    """The detected chessboard, plus what calibration recovered from it."""
    gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    drawn = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if found:
        cv2.drawChessboardCorners(drawn, pattern, corners, found)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].imshow(_rgb(drawn))
    axes[0].set_title(f"Detected corners — {pattern[0]}×{pattern[1]}")
    axes[0].axis("off")

    axes[1].axis("off")
    text = (
        f"Zhang calibration\n\n"
        f"views used       {calib.n_images}\n"
        f"RMS reprojection {calib.rms:.4f} px\n\n"
        f"fx  {calib.fx:10.2f}\n"
        f"fy  {calib.fy:10.2f}\n"
        f"cx  {calib.cx:10.2f}\n"
        f"cy  {calib.cy:10.2f}\n\n"
        f"distortion\n"
        f"k1 {calib.dist[0]: .4f}   k2 {calib.dist[1]: .4f}\n"
        f"p1 {calib.dist[2]: .4f}   p2 {calib.dist[3]: .4f}\n"
        f"k3 {calib.dist[4]: .4f}"
    )
    axes[1].text(0.02, 0.98, text, va="top", family="monospace", fontsize=10)
    fig.suptitle("Stage 1 — camera calibration")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rectification(left, right, stats, path, n_lines=14):
    """
    The pair side by side with horizontal rules drawn across both.

    This is the figure that makes rectification checkable by eye: on a
    rectified pair every feature sits on the same rule in both halves. The
    measured vertical disparity is printed alongside, because a few pixels of
    misalignment is easy to miss visually and fatal to block matching.
    """
    pair = np.hstack([_rgb(left), _rgb(right)])
    h, w = left.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(pair)
    for y in np.linspace(h * 0.05, h * 0.95, n_lines):
        ax.axhline(y, color="#ff3030", linewidth=0.6, alpha=0.75)
    ax.axvline(w, color="white", linewidth=1.5)
    ax.set_xticks([]), ax.set_yticks([])

    if "mean_abs_dy" in stats:
        verdict = "RECTIFIED" if stats["rectified"] else "NOT RECTIFIED"
        ax.set_title(
            f"Stage 2 — rectification check: {verdict}\n"
            f"mean |Δy| = {stats['mean_abs_dy']:.2f} px over "
            f"{stats['n_inliers']} matches   ·   disparity "
            f"{stats['disparity_min']:.1f}–{stats['disparity_max']:.1f} px",
            fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_disparity(left, disparity, path, method="sgbm"):
    """The left image, the disparity map, and where it has no answer."""
    valid = np.isfinite(disparity)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].imshow(_rgb(left))
    axes[0].set_title("Left image")

    shown = np.ma.masked_where(~valid, disparity)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#101010")
    im = axes[1].imshow(shown, cmap=cmap)
    axes[1].set_title(f"Disparity ({method.upper()}), pixels")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(valid, cmap="gray")
    axes[2].set_title(f"Valid matches — {valid.mean():.1%} of pixels")

    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Stage 3 — disparity")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_method_comparison(left, maps, path):
    """
    Block matching against semi-global matching on the same pair.

    The contrast is the argument for the default: BM decides every pixel on
    its own and gives up wherever the surface has no texture, which on an
    indoor scene is most of it.
    """
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(5 * (len(maps) + 1), 4.4))
    axes[0].imshow(_rgb(left))
    axes[0].set_title("Left image")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#101010")
    for ax, (name, d) in zip(axes[1:], maps.items()):
        valid = np.isfinite(d)
        ax.imshow(np.ma.masked_where(~valid, d), cmap=cmap)
        ax.set_title(f"{name.upper()} — {valid.mean():.1%} valid")

    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Block matching vs semi-global matching")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_depth(depth, path, baseline_known, unit):
    """The depth map and the distribution behind it."""
    valid = np.isfinite(depth)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#101010")
    im = axes[0].imshow(np.ma.masked_where(~valid, depth), cmap=cmap)
    axes[0].set_title(f"Depth ({unit}) — near is light")
    axes[0].set_xticks([]), axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].hist(depth[valid], bins=80, color="#2b6cb0")
    axes[1].set_xlabel(f"depth ({unit})")
    axes[1].set_ylabel("pixels")
    axes[1].set_title("Depth distribution")

    note = ("" if baseline_known else
            "  —  baseline not measured, so depth is relative, not metric")
    fig.suptitle(f"Stage 4 — depth, Z = fx·B/d{note}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_point_cloud(points, colors, path, unit):
    """
    The reconstructed cloud from three viewpoints.

    The top-down view is the useful one: the scene's layout in depth is what
    a disparity map is for, and it is invisible from the camera's own angle.
    """
    if len(points) == 0:
        return None

    # Thinning only the figure, never the saved cloud: a 100k-point scatter
    # renders slowly and hides its own structure under overdraw.
    if len(points) > 60000:
        idx = np.random.default_rng(0).choice(len(points), 60000, replace=False)
        points, colors = points[idx], colors[idx]

    views = [((20, -70), "oblique"), ((89, -90), "top-down"), ((0, -90), "front")]
    fig = plt.figure(figsize=(16, 5.2))
    for k, ((elev, azim), name) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(points[:, 0], points[:, 2], -points[:, 1],
                   c=colors, s=0.6, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_xlabel("X"), ax.set_ylabel("Z"), ax.set_zlabel("-Y")
        ax.set_box_aspect([1, 1, 1])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticklabels([])
    fig.suptitle(f"Stage 5 — point cloud, {len(points):,} points "
                 f"(depth in {unit})")
    # The 3D axes claim the full subplot, so tight_layout alone lets the
    # per-view titles ride up into the suptitle.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_disparity_image(disparity, path):
    """The disparity as a plain 8-bit PNG, normalised for viewing only."""
    cv2.imwrite(str(path), disp_mod.for_display(disparity))
    return path
