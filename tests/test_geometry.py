"""
tests/test_geometry.py
========================
Unit tests for geometry conversion functions.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.utils.geometry import (
    px_to_cm_horizontal,
    px_to_cm_vertical,
    cm_to_px_horizontal,
    ramanujan_perimeter,
    z_from_a4_paper,
    ankle_floor_correction_px,
)


# ── Focal length constants used in tests ────────────────────────────
FX = 934.2
FY = 935.7
Z  = 200.0  # cm


class TestPixelCmConversion:
    def test_horizontal_round_trip(self):
        cm  = 10.0
        px  = cm_to_px_horizontal(cm, Z, FX)
        out = px_to_cm_horizontal(px, Z, FX)
        assert abs(out - cm) < 0.001

    def test_vertical_at_known_z(self):
        """At Z=fy, one pixel should equal exactly 1 cm."""
        result = px_to_cm_vertical(1.0, FY, FY)
        assert abs(result - 1.0) < 1e-6

    def test_larger_z_gives_larger_cm(self):
        """Same pixel span appears larger at greater depth."""
        cm_near = px_to_cm_horizontal(100, 150.0, FX)
        cm_far  = px_to_cm_horizontal(100, 300.0, FX)
        assert cm_far > cm_near


class TestRamanujanPerimeter:
    def test_circle(self):
        """For a=b (circle), perimeter should equal 2*pi*a."""
        a = 5.0
        expected = 2 * math.pi * a
        result   = ramanujan_perimeter(a, a)
        assert abs(result - expected) < 0.001

    def test_known_ellipse(self):
        """a=6, b=4 — known perimeter ≈ 32.66 cm."""
        result = ramanujan_perimeter(6.0, 4.0)
        assert 32.0 < result < 33.5

    def test_symmetry(self):
        """Perimeter should be the same regardless of which axis is a vs b."""
        p1 = ramanujan_perimeter(3.0, 5.0)
        p2 = ramanujan_perimeter(5.0, 3.0)
        assert abs(p1 - p2) < 1e-6

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            ramanujan_perimeter(0, 5)
        with pytest.raises(ValueError):
            ramanujan_perimeter(-1, 5)


class TestZFromA4Paper:
    def test_z_reasonable_range(self):
        """At 2m distance, A4 paper (21cm wide) takes ~fx/Z*21 ≈ 98 px."""
        Z_true   = 200.0
        paper_w  = FX * 21.0 / Z_true
        paper_h  = FY * 29.7 / Z_true
        z_est    = z_from_a4_paper(paper_w, paper_h, FX, FY)
        assert abs(z_est - Z_true) < 1.0

    def test_closer_gives_smaller_z(self):
        z_near = z_from_a4_paper(200, 282, FX, FY)   # larger pixels = closer
        z_far  = z_from_a4_paper(100, 141, FX, FY)
        assert z_near < z_far


class TestAnkleFloorCorrection:
    def test_positive_correction(self):
        """Correction should always push foot_y downward (positive pixels)."""
        corr = ankle_floor_correction_px(3.5, 200.0, FY)
        assert corr > 0

    def test_closer_gives_larger_correction_px(self):
        """At smaller Z, same physical offset = more pixels."""
        corr_near = ankle_floor_correction_px(3.5, 150.0, FY)
        corr_far  = ankle_floor_correction_px(3.5, 250.0, FY)
        assert corr_near > corr_far
