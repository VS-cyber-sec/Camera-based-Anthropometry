"""
Utility modules for camera-based anthropometry.

    geometry  — pixel/cm conversions, Ramanujan perimeter, Z from A4
    filters   — IQR filter, robust estimator, EMA smoother
"""
from .geometry import (
    px_to_cm_horizontal,
    px_to_cm_vertical,
    cm_to_px_horizontal,
    cm_to_px_vertical,
    ramanujan_perimeter,
    z_from_a4_paper,
    ankle_floor_correction_px,
)
from .filters import iqr_filter, robust_estimate, exponential_moving_average

__all__ = [
    "px_to_cm_horizontal", "px_to_cm_vertical",
    "cm_to_px_horizontal", "cm_to_px_vertical",
    "ramanujan_perimeter", "z_from_a4_paper",
    "ankle_floor_correction_px",
    "iqr_filter", "robust_estimate", "exponential_moving_average",
]
