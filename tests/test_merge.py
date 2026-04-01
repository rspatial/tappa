"""
Tests ported from inst/tinytest/test_merge.R

Covers merge, mosaic, and blend.
"""
import numpy as np
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa.rast import rast
from tappa.values import set_values, values as get_values
from tappa.merge import merge, mosaic, blend
from tappa.names import set_names_inplace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_two_layer(xmin, xmax, val):
    """Create a 2-layer raster covering the given x range."""
    r = rast(nrows=180, ncols=360, xmin=xmin, xmax=xmax, ymin=-90, ymax=90)
    r = set_values(r, float(val))
    opt = pt.SpatOptions()
    r.addSource(r.deepcopy(), True, opt)
    return r


def _vals(r) -> np.ndarray:
    """Read all values as a flat float64 array."""
    r.readStart()
    try:
        v = np.array(r.readValues(0, r.nrow(), 0, r.ncol()), dtype=float)
    finally:
        r.readStop()
    return v


def _vals_2d(r) -> np.ndarray:
    """Read values as a (nrow, ncol) matrix (single-layer)."""
    v = _vals(r)
    return v.reshape(int(r.nrow()), int(r.ncol()))


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

class TestMerge:

    def test_merge_layer_names_preserved(self):
        """Names from the first raster are preserved in the merge result."""
        r1 = _make_two_layer(0, 1, 1)
        r2 = _make_two_layer(1, 2, 3)
        set_names_inplace(r1, ["x", "y"])
        set_names_inplace(r2, ["x", "y"])
        m = merge(r1, r2)
        assert list(m.names) == ["x", "y"]


# ---------------------------------------------------------------------------
# mosaic
# ---------------------------------------------------------------------------

class TestMosaic:

    def _make_xyz(self):
        """Three overlapping rasters with constant values 1, 2, 3."""
        x = rast(xmin=-110, xmax=-60, ymin=40, ymax=70, ncols=50, nrows=30)
        x = set_values(x, 1.0)
        y = rast(xmin=-95, xmax=-45, ymin=30, ymax=60, ncols=50, nrows=30)
        y = set_values(y, 2.0)
        z = rast(xmin=-80, xmax=-30, ymin=20, ymax=50, ncols=50, nrows=30)
        z = set_values(z, 3.0)
        return x, y, z

    def test_mosaic_mean_diagonal(self):
        """Diagonal of mosaic(x,y,z, fun='mean') matches R output."""
        x, y, z = self._make_xyz()
        m = mosaic(x, y, z, fun="mean")
        mat = _vals_2d(m)

        # diagonal of the output matrix
        d = np.diag(mat)

        # In non-overlapping zone of x (top-left): value = 1
        # In x∩y overlap: mean(1,2) = 1.5
        # In y∩z overlap: mean(2,3) = 2.5
        # In z-only zone (bottom-right): value = 3
        # The exact expected values from R:
        expected = np.array([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5,
            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
            2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        ])

        np.testing.assert_array_almost_equal(d, expected)

    def test_mosaic_sum(self):
        """In overlapping area of two val=1 rasters, sum should be 2."""
        r1 = rast(xmin=0, xmax=10, ymin=0, ymax=10, ncols=10, nrows=10)
        r1 = set_values(r1, 1.0)
        r2 = rast(xmin=5, xmax=15, ymin=0, ymax=10, ncols=10, nrows=10)
        r2 = set_values(r2, 1.0)
        m = mosaic(r1, r2, fun="sum")
        mat = _vals_2d(m)
        # cols 0-4: only r1 → 1, cols 5-9: overlap → 2, cols 10-14: only r2 → 1
        np.testing.assert_array_almost_equal(mat[0, :5], 1.0)
        np.testing.assert_array_almost_equal(mat[0, 5:10], 2.0)
        np.testing.assert_array_almost_equal(mat[0, 10:], 1.0)

    def test_mosaic_min_max(self):
        """min/max mosaic of rasters with values 1 and 3."""
        r1 = rast(xmin=0, xmax=10, ymin=0, ymax=10, ncols=10, nrows=10)
        r1 = set_values(r1, 1.0)
        r2 = rast(xmin=5, xmax=15, ymin=0, ymax=10, ncols=10, nrows=10)
        r2 = set_values(r2, 3.0)

        lo = mosaic(r1, r2, fun="min")
        hi = mosaic(r1, r2, fun="max")
        lo_mat = _vals_2d(lo)
        hi_mat = _vals_2d(hi)

        # overlap cols 5-9
        np.testing.assert_array_almost_equal(lo_mat[0, 5:10], 1.0)
        np.testing.assert_array_almost_equal(hi_mat[0, 5:10], 3.0)


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------

class TestBlend:

    def _make_xyz(self):
        """Same three overlapping rasters as mosaic test."""
        x = rast(xmin=-110, xmax=-60, ymin=40, ymax=70, ncols=50, nrows=30)
        x = set_values(x, 1.0)
        y = rast(xmin=-95, xmax=-45, ymin=30, ymax=60, ncols=50, nrows=30)
        y = set_values(y, 2.0)
        z = rast(xmin=-80, xmax=-30, ymin=20, ymax=50, ncols=50, nrows=30)
        z = set_values(z, 3.0)
        return x, y, z

    def test_blend_diagonal(self):
        """Diagonal of blend(x,y,z) matches R output."""
        x, y, z = self._make_xyz()
        m = blend(x, y, z)
        mat = _vals_2d(m)
        d = np.diag(mat)

        # Expected values from R test_merge.R
        expected = np.array([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1.03333333333333, 1.1, 1.16666666666667,
            1.23333333333333, 1.3, 1.36666666666667,
            1.43333333333333, 1.5, 1.56666666666667,
            1.63333333333333, 1.7, 1.76666666666667,
            1.83333333333333, 1.88461538461538, 1.95454545454545,
            2.05, 2.15, 2.25, 2.35, 2.45,
            2.55, 2.65, 2.75, 2.85, 2.95,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        ])

        np.testing.assert_array_almost_equal(d, expected, decimal=5)

    def test_blend_smooth_transition(self):
        """Values in the overlap zone are strictly between the two inputs."""
        r1 = rast(xmin=0, xmax=10, ymin=0, ymax=10, ncols=20, nrows=20)
        r1 = set_values(r1, 0.0)
        r2 = rast(xmin=5, xmax=15, ymin=0, ymax=10, ncols=20, nrows=20)
        r2 = set_values(r2, 10.0)

        m = blend(r1, r2)
        mat = _vals_2d(m)

        # overlap columns (roughly cols 10-19 in the output, which is 30 cols)
        # Values should transition smoothly from 0 to 10
        mid_row = mat[10, :]
        overlap = mid_row[~np.isnan(mid_row)]

        # should start near 0 and end near 10
        assert overlap[0] == pytest.approx(0.0, abs=0.1)
        assert overlap[-1] == pytest.approx(10.0, abs=0.1)

        # should be monotonically non-decreasing
        diffs = np.diff(overlap)
        assert np.all(diffs >= -1e-10)

    def test_blend_order_independent(self):
        """Blend result is the same regardless of input order."""
        r1 = rast(xmin=0, xmax=10, ymin=0, ymax=10, ncols=10, nrows=10)
        r1 = set_values(r1, 1.0)
        r2 = rast(xmin=5, xmax=15, ymin=0, ymax=10, ncols=10, nrows=10)
        r2 = set_values(r2, 2.0)
        r3 = rast(xmin=10, xmax=20, ymin=0, ymax=10, ncols=10, nrows=10)
        r3 = set_values(r3, 3.0)

        m_abc = blend(r1, r2, r3)
        m_cba = blend(r3, r2, r1)
        m_bac = blend(r2, r1, r3)

        v_abc = _vals(m_abc)
        v_cba = _vals(m_cba)
        v_bac = _vals(m_bac)

        np.testing.assert_array_almost_equal(v_abc, v_cba)
        np.testing.assert_array_almost_equal(v_abc, v_bac)

    def test_blend_non_overlapping_equals_merge(self):
        """When rasters don't overlap, blend produces the same result as merge."""
        r1 = rast(xmin=0, xmax=5, ymin=0, ymax=10, ncols=5, nrows=10)
        r1 = set_values(r1, 1.0)
        r2 = rast(xmin=5, xmax=10, ymin=0, ymax=10, ncols=5, nrows=10)
        r2 = set_values(r2, 2.0)

        b = blend(r1, r2)
        m = merge(r1, r2)
        np.testing.assert_array_almost_equal(_vals(b), _vals(m))

    def test_blend_single_raster(self):
        """Blend of a single raster returns it unchanged."""
        r = rast(xmin=0, xmax=10, ymin=0, ymax=10, ncols=10, nrows=10)
        r = set_values(r, 42.0)
        b = blend(r)
        np.testing.assert_array_almost_equal(_vals(b), _vals(r))
