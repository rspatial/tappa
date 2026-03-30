"""
Tests ported from inst/tinytest/test_misc-raster.R

Value checks for raster spatial-context and distance helpers (see terra man/*.Rd).
Tests that depend on APIs not yet exposed in tappa are marked skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa.rast import rast
from tappa.values import set_values
from tappa.generics import (
    boundaries,
    patches,
    cellSize,
    surfArea,
    terrain,
    shade,
    nidp,
    sieve,
)
from tappa.distance import cost_dist, grid_dist
from tappa.stats import autocor
from tappa.subset import subset_rast

from path_utils import skip_if_missing_inst_ex


def _first_layer(r):
    """Single-layer raster (layer 0) when *r* has multiple layers."""
    if r.nlyr() > 1:
        return subset_rast(r, 0)
    return r


def _vals(r):
    r.readStart()
    try:
        return np.array(r.readValues(0, r.nrow(), 0, r.ncol()), dtype=float)
    finally:
        r.readStop()


def _cell_value(r, cell_1based: int):
    """Extract single cell by 1-based cell index (R convention)."""
    v = _vals(r)
    return float(v[cell_1based - 1])


def test_adjacent_rook_center_1x5():
    """R cell 3 → neighbours 2 and 4 (1-based); C++ uses 0-based cell indices."""
    ra = rast(nrows=1, ncols=5, xmin=0, xmax=5, ymin=0, ymax=1, crs="local")
    ra = set_values(ra, np.arange(1, 6, dtype=float))
    adj = ra.adjacent([2.0], "rook", False)
    adj_arr = np.asarray(adj, dtype=float)
    neighbors = sorted(int(x) for x in adj_arr[np.isfinite(adj_arr)])
    assert neighbors == [1, 3]


def test_adjacent_queen_center_3x3():
    """Centre cell (index 4, 0-based) of a 3×3 queen grid → 8 neighbours."""
    ra = rast(nrows=3, ncols=3, xmin=0, xmax=3, ymin=0, ymax=3, crs="local")
    ra = set_values(ra, np.arange(1, 10, dtype=float))
    adj = ra.adjacent([4.0], "queen", False)
    adj_arr = np.asarray(adj, dtype=float)
    neighbors = sorted(int(x) for x in adj_arr[np.isfinite(adj_arr)])
    assert neighbors == [0, 1, 2, 3, 5, 6, 7, 8]


def test_boundaries_detects_edges():
    """Port of R: boundaries(rb, classes=TRUE) on 1x5 raster [1,1,2,2,2] → [0,1,1,0,0]."""
    rb = rast(nrows=1, ncols=5, xmin=0, xmax=5, ymin=0, ymax=1, crs="local")
    rb = set_values(rb, np.array([1.0, 1.0, 2.0, 2.0, 2.0]))
    b = boundaries(rb, classes=True)
    bv = _vals(b)
    np.testing.assert_array_equal(bv, [0.0, 1.0, 1.0, 0.0, 0.0])


def test_patches_labels_connected_regions():
    """Port of R: 2x2 clump of 1s on NA background → single patch, all 4 cells == 1."""
    rp = rast(nrows=4, ncols=4, xmin=0, xmax=4, ymin=0, ymax=4, crs="local")
    vals = np.full(16, float("nan"))
    # rows 1-2, cols 1-2 in 0-based → indices 5,6,9,10
    vals[5] = vals[6] = vals[9] = vals[10] = 1.0
    rp = set_values(rp, vals)
    pp = patches(rp)
    pv = _vals(pp)
    assert int(np.sum(pv[np.isfinite(pv)] == 1)) == 4


def test_cell_size_positive():
    """Port of R: cellSize > 0 and sum of cell sizes > 0 (elev.tif)."""
    f = skip_if_missing_inst_ex("elev.tif")
    r = _first_layer(rast(f))
    cs = cellSize(r)
    csv = _vals(cs)
    assert np.all(csv[np.isfinite(csv)] > 0)
    assert float(np.nansum(csv)) > 0


def test_surf_area_positive():
    """Port of R: surfArea on projected (UTM) raster → all values > 0."""
    rpj = rast(
        nrows=10, ncols=10,
        crs="+proj=utm +zone=32 +datum=WGS84",
        xmin=0, xmax=1000, ymin=0, ymax=1000,
    )
    rpj = set_values(rpj, np.random.uniform(size=100))
    sa = surfArea(rpj)
    sav = _vals(sa)
    assert np.all(sav[np.isfinite(sav)] > 0)


def test_cost_dist_nonneg():
    """Port of R: costDist/gridDist with target=0; source cell distance == 0."""
    rc = rast(
        ncols=5, nrows=5,
        crs="+proj=utm +zone=1 +datum=WGS84",
        xmin=0, xmax=5, ymin=0, ymax=5,
    )
    rc = set_values(rc, np.ones(25))
    # cell index 12 (0-based) = cell 13 in R (1-based) → set to 0 (source)
    vals = np.ones(25)
    vals[12] = 0.0
    rc = set_values(rc, vals)
    cd = cost_dist(rc, target=0.0)
    cdv = _vals(cd)
    # source cell distance should be 0
    assert cdv[12] == pytest.approx(0.0)


def test_autocor_global_moran_bounds():
    """Global Moran's I for a raster should be in (-1, 1)."""
    f = skip_if_missing_inst_ex("elev.tif")
    r = _first_layer(rast(f))
    aci = autocor(r, global_=True, method="moran")
    assert isinstance(aci, float)
    assert -1.0 <= aci <= 1.0


def test_terrain_slope_aspect_on_elev():
    """Port of R: shade(slope, asp); min >= 0 and max <= 1 (na.rm=TRUE)."""
    f = skip_if_missing_inst_ex("elev.tif")
    r = _first_layer(rast(f))
    slope = terrain(r, v="slope", unit="radians")
    asp = terrain(r, v="aspect", unit="radians")
    sh = shade(slope, asp)
    v = _vals(sh)
    finite = v[np.isfinite(v)]
    assert len(finite) > 0
    assert float(finite.min()) >= 0.0
    assert float(finite.max()) <= 1.0


def test_nidp_flowdir():
    """Port of R: NIDP(terrain(r, 'flowdir')) → all values in [0, 9]."""
    f = skip_if_missing_inst_ex("elev.tif")
    r = _first_layer(rast(f))
    fd = terrain(r, v="flowdir")
    nid = nidp(fd)
    v = _vals(nid)
    finite = v[np.isfinite(v)]
    assert len(finite) > 0
    assert np.all(finite >= 0)
    assert np.all(finite <= 9)
