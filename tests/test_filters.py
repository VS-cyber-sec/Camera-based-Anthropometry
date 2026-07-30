"""
tests/test_filters.py
=======================
Unit tests for IQR filter and robust estimator.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.utils.filters import iqr_filter, robust_estimate, exponential_moving_average


class TestIQRFilter:
    def test_removes_high_outlier(self):
        vals   = [160.0, 161.0, 160.5, 159.5, 160.2, 500.0]
        result = iqr_filter(vals)
        assert 500.0 not in result

    def test_removes_low_outlier(self):
        vals   = [30.0, 29.5, 30.2, 29.8, 30.1, 1.0]
        result = iqr_filter(vals)
        assert 1.0 not in result

    def test_keeps_all_valid(self):
        vals   = [160.0, 161.0, 160.5, 159.5, 160.2]
        result = iqr_filter(vals)
        assert len(result) == len(vals)

    def test_short_list_unchanged(self):
        vals   = [160.0, 161.0, 162.0]
        result = iqr_filter(vals)
        assert result == vals


class TestRobustEstimate:
    def test_median_of_clean_data(self):
        vals   = [160.0, 161.0, 160.5, 159.5, 160.2]
        result = robust_estimate(vals)
        assert result is not None
        assert 159 < result < 162

    def test_outlier_removed_before_median(self):
        vals         = [160.0] * 10 + [300.0]
        with_outlier = robust_estimate(vals)
        assert with_outlier is not None
        assert abs(with_outlier - 160.0) < 1.0

    def test_empty_returns_none(self):
        assert robust_estimate([]) is None

    def test_returns_rounded_to_1dp(self):
        vals   = [160.123, 160.456, 160.789]
        result = robust_estimate(vals)
        # Should be rounded to 1 decimal place
        assert result == round(result, 1)


class TestEMA:
    def test_first_call_returns_value(self):
        result = exponential_moving_average(200.0, None)
        assert result == 200.0

    def test_smoothing(self):
        est = 200.0
        for _ in range(10):
            est = exponential_moving_average(210.0, est, alpha=0.25)
        # Should move toward 210 but not reach it in 10 steps
        assert 205 < est < 210

    def test_large_jump_damped(self):
        est    = 200.0
        result = exponential_moving_average(300.0, est,
                                            alpha=0.25, max_jump=30.0)
        # 300 is a 100cm jump (>30cm threshold) — should be damped
        assert result < 250.0
        assert result > 200.0
