"""
Writing the point cloud out.

Binary little-endian PLY with `x y z red green blue`, which is the layout
MeshLab, CloudCompare, Open3D, Blender and the Windows 3D Viewer all read
without conversion. Written by hand so the project has no hard dependency on
Open3D just to save a file.
"""

import json

import numpy as np


def write_ply(path, points, colors=None):
    """
    Write a coloured point cloud.

    :param points: (n, 3)
    :param colors: (n, 3) RGB in 0..1; mid-grey is used when omitted
    :returns: (path, number of points)
    """
    points = np.asarray(points, np.float32)
    n = len(points)
    if colors is None:
        colors = np.full((n, 3), 0.5)
    rgb = (np.clip(np.asarray(colors), 0.0, 1.0) * 255.0).astype(np.uint8)

    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(b"comment generated from a stereo disparity map\n")
        f.write(f"element vertex {n}\n".encode())
        for name in ("x", "y", "z"):
            f.write(f"property float {name}\n".encode())
        for name in ("red", "green", "blue"):
            f.write(f"property uchar {name}\n".encode())
        f.write(b"end_header\n")

        rows = np.empty(n, dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
        rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
        rows["red"], rows["green"], rows["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rows.tobytes())
    return path, n


def write_xyz(path, points, colors=None):
    """The same cloud as plain text, one point per line."""
    points = np.asarray(points, float)
    if colors is None:
        colors = np.full((len(points), 3), 0.5)
    rgb = (np.clip(np.asarray(colors), 0, 1) * 255).astype(int)
    with open(path, "w") as f:
        for p, c in zip(points, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{c[0]} {c[1]} {c[2]}\n")
    return path, len(points)


def write_stats(stats, path):
    """The run's numbers, machine-readable."""
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path
