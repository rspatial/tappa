"""
Tests ported from inst/tinytest/test_raster-local.R

Value checks for local / cell-based raster functions.
All R tests are now covered; the six functions that were previously
missing (roll, thresh, extractRange, selectHighest, divide, approximate)
have been implemented in tappa.generics and are tested here.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa._terra import SpatOptions
from tappa.rast import rast
from tappa.values import set_values
from tappa.arith import not_na, which_lyr, as_bool_rast, rast_modal
from tappa.generics import (
    clamp, subst, cover, diff_raster, segregate,
    classify, selectRange, scale_linear,
    roll, thresh, select_highest, divide, approximate, extract_range,
)
from tappa.init import init
from tappa.focal import focal_mat
from tappa.app import app
from tappa.zonal import zonal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vals(r) -> np.ndarray:
    """Read all raster values as a flat float64 array."""
    r.readStart()
    try:
        v = np.array(r.readValues(0, r.nrow(), 0, r.ncol()), dtype=float)
    finally:
        r.readStop()
    return v


def _stack(*rasters):
    """Combine SpatRasters into a multi-layer raster (like R c(r1, r2, ...))."""
    opt = SpatOptions()
    out = rasters[0].deepcopy()
    for r in rasters[1:]:
        out.addSource(r, True, opt)
    return out


def _make_r():
    """
    9×9 raster from (0,1)×(0,1) with values 1..81 — mirrors R:
        r <- rast(nrows=9, ncols=9, xmin=0, xmax=1, ymin=0, ymax=1)
        values(r) <- 1:ncell(r)
    """
    r = rast(nrows=9, ncols=9, xmin=0, xmax=1, ymin=0, ymax=1)
    r = set_values(r, list(range(1, 82)))
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clamp_bounds():
    """clamp(r, lower=10, upper=50) matches pmin(pmax(v, 10), 50)."""
    r = _make_r()
    v = _vals(r)
    cl = clamp(r, lower=10, upper=50)
    expected = np.clip(v, 10, 50)
    np.testing.assert_array_almost_equal(_vals(cl), expected)


def test_subst_to_na():
    """subst(r, from=[1,2], to=[NA,NA]) — values 1 and 2 become NaN."""
    r = _make_r()
    v = _vals(r)
    sb = subst(r, from_val=[1.0, 2.0], to_val=[float("nan"), float("nan")])
    sv = _vals(sb)
    mask_na = np.isin(v, [1.0, 2.0])
    assert np.all(np.isnan(sv[mask_na]))
    np.testing.assert_array_almost_equal(sv[~mask_na], v[~mask_na])


def test_init_cell_sequential():
    """init(r, 'cell') returns sequential cell indices covering all ncell values."""
    r = _make_r()
    ini = init(r, "cell")
    vals = _vals(ini)
    ncell = r.nrow() * r.ncol()
    assert len(vals) == ncell
    assert np.all(np.diff(vals) == 1.0), "init('cell') values are not sequential"
    assert vals[0] in (0.0, 1.0), f"Unexpected first cell index: {vals[0]}"
    assert vals[-1] == vals[0] + ncell - 1


def test_segregate_two_classes():
    """segregate(r < 20): 2 unique values → 2 layers, all ones sum == ncell."""
    r = _make_r()
    sg = segregate(r < 20)
    assert sg.nlyr() == 2, f"Expected 2 layers, got {sg.nlyr()}"
    v = _vals(sg)
    ncell = r.nrow() * r.ncol()
    valid = v[~np.isnan(v)]
    assert int(np.sum(valid)) == ncell


def test_cover_noNA_unchanged():
    """cover(r, r2) where r has no NA: result equals r."""
    r = _make_r()
    r2 = r * 2
    v = _vals(r)
    cv = cover(r, r2)
    np.testing.assert_array_almost_equal(_vals(cv), v)


def test_roll_constant_mean():
    """
    roll on constant 5-layer stack: rolling mean (n=3) equals constant 1.
    Mirrors R: roll(stk, 3, 'mean', circular=TRUE).
    """
    r1 = rast(nrows=3, ncols=3, xmin=0, xmax=1, ymin=0, ymax=1)
    r1 = set_values(r1, [1.0] * 9)
    stk = _stack(r1, r1, r1, r1, r1)
    rl = roll(stk, n=3, fun="mean", circular=True)
    v = _vals(rl)
    valid = v[~np.isnan(v)]
    assert len(valid) > 0
    assert np.max(np.abs(valid - 1.0)) < 1e-10


def test_diff_second_minus_first():
    """diff_raster(c(r, r*2)) == r (layer 2 - layer 1 = 2r - r = r)."""
    r = _make_r()
    v = _vals(r)
    stk = _stack(r, r * 2)
    d = diff_raster(stk, lag=1)
    np.testing.assert_array_almost_equal(_vals(d), v)


def test_rast_modal_identical_layers():
    """modal(c(r, r, r)): modal of three identical layers equals r."""
    r = _make_r()
    v = _vals(r)
    stk = _stack(r, r, r)
    md = rast_modal(stk)
    np.testing.assert_array_almost_equal(_vals(md), v)


def test_thresh_mean_split():
    """
    thresh(r, method='mean'): result equals (r > mean(r)).
    Mirrors R: br <- ifel(r > mu, 1, 0); expect_equal(values(tr), values(br)).
    """
    r = _make_r()
    v = _vals(r)
    mu = float(np.mean(v))
    tr = thresh(r, method="mean", as_raster=True)
    tr_v = _vals(tr)
    expected = (v > mu).astype(float)
    np.testing.assert_array_almost_equal(tr_v, expected)


def test_scale_linear_unit_range():
    """scale_linear(r, min=0, max=1) maps to [0, 1] via (v-min)/(max-min)."""
    r = _make_r()
    v = _vals(r)
    scl = scale_linear(r, min=0.0, max=1.0)
    mn, mx = v.min(), v.max()
    expected = (v - mn) / (mx - mn)
    np.testing.assert_array_almost_equal(_vals(scl), expected)


def test_focal_mat_circle_7x7():
    """
    focalMat(r, d=3*res, type='circle') → 7×7 for res=1/9 and d=3*(1/9).
    In tappa focal_mat takes radius in cells: radius = d/res = 3.
    """
    fm = focal_mat("circle", r=3)
    assert fm.shape == (7, 7), f"Expected (7,7), got {fm.shape}"
    assert np.nansum(fm) > 0


def test_which_lyr_first_always_true():
    """
    which.lyr(c(r>0, r>10000)): all r values are 1..81 so layer 0 (0-based)
    is always the first TRUE layer.
    """
    r = _make_r()
    stk = _stack(r > 0, r > 10000)
    wly = which_lyr(stk)
    vals = _vals(wly)
    valid = vals[~np.isnan(vals)]
    assert np.all(valid == 0), f"Expected all zeros, got unique={np.unique(valid)}"


def test_selectRange_by_classify():
    """
    selectRange(c(r, r*2, r*3), idx):
      idx==1 → layer 1 (r values), idx==2 → layer 2 (r*2 values).
    """
    r = _make_r()
    v = _vals(r)
    rcl = [[-np.inf, 40.0, 1.0], [40.0, np.inf, 2.0]]
    idx = classify(r, rcl)
    idx_v = _vals(idx)
    stk3 = _stack(r, r * 2, r * 3)
    sr = selectRange(stk3, idx)
    sr_v = _vals(sr)
    expected = np.where(idx_v == 1, v, 2.0 * v)
    np.testing.assert_array_almost_equal(sr_v, expected)


def test_selectHighest_five_cells():
    """select_highest(r, n=5): exactly 5 cells marked 1, rest NA."""
    r = _make_r()
    sh = select_highest(r, n=5)
    v = _vals(sh)
    assert int(np.nansum(v == 1)) == 5, (
        f"Expected 5 cells == 1, got {int(np.nansum(v == 1))}"
    )
    # All non-selected cells are NaN (R: x <- rast(x); x[i] <- 1)
    assert int(np.sum(~np.isnan(v))) == 5


def test_classify_2col_lookup():
    """
    classify with 2-column lookup: from=1:3, to=11:13.
    Values 4 and 5 pass through unchanged.
    """
    np.random.seed(1)
    raw = np.random.choice(np.arange(1, 6), size=100, replace=True).astype(float)
    rcin = rast(nrows=10, ncols=10, xmin=0, xmax=1, ymin=0, ymax=1)
    rcin = set_values(rcin, raw.tolist())
    rcl = [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]
    rco = classify(rcin, rcl)
    vi = raw
    vo = _vals(rco)
    np.testing.assert_array_equal(vo[vi == 1], np.full(np.sum(vi == 1), 11.0))
    np.testing.assert_array_equal(vo[vi == 2], np.full(np.sum(vi == 2), 12.0))
    np.testing.assert_array_equal(vo[vi == 3], np.full(np.sum(vi == 3), 13.0))
    np.testing.assert_array_equal(vo[vi > 3], vi[vi > 3])


def test_not_na_no_missing():
    """not_na(r): sum == ncell when raster has no NA values."""
    r = _make_r()
    nn = not_na(r)
    v = _vals(nn)
    ncell = r.nrow() * r.ncol()
    assert int(np.sum(v)) == ncell


def test_as_bool_above_threshold():
    """as_bool_rast(r > 40): sum matches number of cells where value > 40."""
    r = _make_r()
    v = _vals(r)
    ab = as_bool_rast(r > 40)
    v_bool = _vals(ab)
    expected_count = int(np.sum(v > 40))
    assert int(np.nansum(v_bool)) == expected_count


def test_divide_zonal_sum_conserved():
    """
    divide(r, n=2, as_raster=True): zonal sum across all zones equals
    total sum of r, and there are at least 2 zones.
    Mirrors R: zs <- zonal(r, dv, sum, na.rm=TRUE); sum(zs[,2]) == sum(v).
    """
    r = _make_r()
    v = _vals(r)
    dv = divide(r, n=2, as_raster=True)
    # zonal sum: for each zone in dv, sum r values
    zs = zonal(r, dv, fun="sum", na_rm=True)
    # zs is a DataFrame with columns [zone, lyr.1]
    zone_sums = zs.iloc[:, 1].to_numpy(dtype=float)
    assert abs(float(np.sum(zone_sums)) - float(np.sum(v))) < 1e-6, (
        f"sum(zonal_sums)={np.sum(zone_sums):.4f} != sum(v)={np.sum(v):.4f}"
    )
    assert len(zone_sums) >= 2, f"Expected >= 2 zones, got {len(zone_sums)}"


def test_approximate_fills_na():
    """
    approximate(c(ra, rb)): the single NA in ra (cell 5) is filled
    by linear interpolation from the neighbouring layer.
    Mirrors R: ra[5] <- NA; ap <- approximate(c(ra, rb)); !is.na(values(ap)[5,1]).
    """
    ra = rast(ncols=3, nrows=3, xmin=0, xmax=1, ymin=0, ymax=1)
    ra = set_values(ra, list(range(1, 10)))
    rb = ra * 1.1
    # Set cell 4 (0-based) = cell 5 in R (1-based) to NA
    ra_vals = list(range(1, 10))
    ra_vals[4] = float("nan")
    ra = set_values(ra, ra_vals)

    stk = _stack(ra, rb)
    ap = approximate(stk)
    ap_vals = _vals(ap)
    # First-layer values: the (3*3) × 2 BSQ layout; first 9 values are layer 0
    lyr0 = ap_vals[:9]
    assert not np.isnan(lyr0[4]), (
        f"Cell 4 (layer 0) should be filled by interpolation, got {lyr0[4]}"
    )


def test_app_sum_two_layers():
    """app(c(r, r*2), 'sum') == v + 2*v = 3*v for each cell."""
    r = _make_r()
    v = _vals(r)
    r2 = r * 2
    stk = _stack(r, r2)
    rs = app(stk, "sum")
    np.testing.assert_array_almost_equal(_vals(rs), v + 2.0 * v)


def test_extract_range_per_point():
    """
    extractRange(stk, xy, first=0, last=2): for each point, values from
    layers 0..2.  Cross-check against extract() on the full stack.
    Mirrors R: er[[1]][1,] == extract(stk, xy)[1, -1].
    """
    from tappa.extract import extract as _extract

    r = _make_r()
    stk = _stack(r, r * 2, r * 3)
    # Two points inside the raster extent
    xy = [[0.15, 0.15], [0.50, 0.50]]

    # Full extract reference (ID=False → only value columns)
    e_full = _extract(stk, xy, ID=False)
    expected_row0 = e_full.iloc[0].to_numpy(dtype=float)

    er = extract_range(stk, xy, first=0, last=2)

    # Returns a list of DataFrames, one per point
    assert len(er) == 2
    row0 = er[0].iloc[0].to_numpy(dtype=float)
    assert len(row0) == 3
    # extract_range [0..2] on a 3-layer stack must equal full extract
    np.testing.assert_array_almost_equal(row0, expected_row0)
