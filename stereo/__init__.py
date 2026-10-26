"""
3D reconstruction from a stereo disparity map.

The third of three reconstruction methods in this project, and the only one
that recovers a *dense* result: a depth for almost every pixel, rather than a
few thousand triangulated keypoints. It buys that density with a hard
requirement -- two views whose geometry is already known and rectified --
where Structure from Motion asks only for photographs and works the geometry
out for itself.

The pipeline:

    chessboard images  ->  calibration.py   ->  K, distortion
    stereo pair        ->  rectify.py       ->  verified row correspondence
                       ->  disparity.py     ->  d, in pixels
                       ->  depth.py         ->  Z = fx*B/d, then X and Y
                       ->  export.py        ->  a coloured point cloud

Reference
---------
Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.,
chapter 11 (stereo correspondence) and section 12.2 (triangulation).
Hirschmuller, *Stereo Processing by Semi-Global Matching and Mutual
Information*, PAMI 2008 -- the matcher used by default.
"""

from . import calibration, depth, disparity, export, rectify, visualization

__all__ = ["calibration", "depth", "disparity", "export", "rectify",
           "visualization"]
