"""
geometry.py — Pixel-to-centimetre conversion utilities
=======================================================
All spatial conversions follow the pinhole camera model:

    real_cm = (pixel_span × Z_cm) / focal_length_px

Uses calibrated fx / fy from the camera_params.json loaded by the main script.
Import fx, fy from the main module or pass them explicitly.
"""

import math


def px_to_cm_horizontal(px: float, z_cm: float, fx: float) -> float:
    """Convert a horizontal pixel distance at depth z_cm to centimetres."""
    return px * z_cm / fx


def px_to_cm_vertical(px: float, z_cm: float, fy: float) -> float:
    """Convert a vertical pixel distance at depth z_cm to centimetres."""
    return px * z_cm / fy


def cm_to_px_horizontal(cm: float, z_cm: float, fx: float) -> float:
    """Convert a real-world horizontal cm to pixels at depth z_cm."""
    return cm * fx / z_cm


def cm_to_px_vertical(cm: float, z_cm: float, fy: float) -> float:
    """Convert a real-world vertical cm to pixels at depth z_cm."""
    return cm * fy / z_cm


def ramanujan_perimeter(a: float, b: float) -> float:
    """
    Ellipse perimeter via Ramanujan's second approximation.

    Accurate to better than 0.04% for any a, b.
    Reference: Ramanujan (1914). Q.J. Math 45: 350-372.

    Args:
        a: semi-axis 1 (cm)
        b: semi-axis 2 (cm)

    Returns:
        Perimeter in cm
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Semi-axes must be positive, got a={a}, b={b}")
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def z_from_a4_paper(
    paper_w_px: float,
    paper_h_px: float,
    fx: float,
    fy: float,
    ref_w_cm: float = 21.0,
    ref_h_cm: float = 29.7,
) -> float:
    """
    Compute depth Z from detected A4 paper dimensions.

    Averages Z computed from the width and height of the detected paper
    rectangle to improve robustness against detection noise.

    Args:
        paper_w_px: detected paper width in pixels (shorter side)
        paper_h_px: detected paper height in pixels (longer side)
        fx: horizontal focal length (pixels)
        fy: vertical focal length (pixels)
        ref_w_cm: A4 width = 21.0 cm
        ref_h_cm: A4 height = 29.7 cm

    Returns:
        Z_paper in cm
    """
    z_from_w = (fx * ref_w_cm) / paper_w_px
    z_from_h = (fy * ref_h_cm) / paper_h_px
    return (z_from_w + z_from_h) / 2.0


def ankle_floor_correction_px(ankle_floor_cm: float, z_cm: float, fy: float) -> int:
    """
    Compute the floor correction in pixels for the ankle landmark.

    MediaPipe's heel landmark is at the ankle bone, approximately
    3.5 cm above the actual floor contact point.

    Args:
        ankle_floor_cm: height of ankle bone above floor (default 3.5 cm)
        z_cm: depth to body plane
        fy: vertical focal length

    Returns:
        Pixel offset to add to heel_y to reach floor level
    """
    return int(ankle_floor_cm * fy / z_cm)
