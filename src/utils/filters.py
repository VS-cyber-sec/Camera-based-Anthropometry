"""
filters.py — Robust statistical filters for measurement stabilisation
=====================================================================
"""

import numpy as np
from typing import List, Optional


def iqr_filter(values: List[float], k: float = 1.5) -> List[float]:
    """
    Remove outliers using the Interquartile Range (IQR) method.

    Removes values outside [Q1 - k*IQR, Q3 + k*IQR].
    Falls back to the original list if fewer than 3 values remain after filtering.

    Args:
        values: list of measurement samples
        k: IQR multiplier (default 1.5 — standard Tukey fence)

    Returns:
        Filtered list with outliers removed
    """
    if len(values) < 4:
        return values
    arr = np.array(values, dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    mask = (arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)
    filtered = arr[mask].tolist()
    return filtered if len(filtered) >= 3 else values


def robust_estimate(values: List[float]) -> Optional[float]:
    """
    IQR-filter then take the median of the remaining values.

    This is the final estimator used for all three measurements.
    Robust against outlier frames (posture drift, occlusion spikes).

    Args:
        values: list of raw measurement samples

    Returns:
        Robust estimate in cm, or None if values is empty
    """
    if not values:
        return None
    filtered = iqr_filter(values)
    return round(float(np.median(filtered)), 1)


def exponential_moving_average(
    new_value: float,
    current_estimate: Optional[float],
    alpha: float = 0.25,
    max_jump: float = 30.0,
) -> float:
    """
    Exponential moving average with large-jump dampening.

    Used to smooth the Z_paper depth reading across frames.
    Rejects sudden large jumps (>max_jump cm) by dampening them
    rather than accepting them directly.

    Args:
        new_value: new raw measurement
        current_estimate: current smoothed estimate (None on first call)
        alpha: smoothing factor (0=no update, 1=always use new value)
        max_jump: threshold in cm above which a jump is dampened

    Returns:
        Updated smoothed estimate
    """
    if current_estimate is None:
        return new_value
    if abs(new_value - current_estimate) > max_jump:
        # Large jump: damp rather than accept
        return 0.6 * current_estimate + 0.4 * new_value
    return alpha * new_value + (1 - alpha) * current_estimate
