"""
Tests for expanse() on SpatRaster (rspatial/tappa#1).

Mirrors R terra::expanse for SpatRaster: area covered by non-NA cells,
with unit, transform, byValue, zones, wide and usenames arguments.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

pytest.importorskip("tappa._terra")
pd = pytest.importorskip("pandas")

import tappa as pt
from tappa.rast import rast
from tappa.values import setValues
from tappa.spatvec import expanse


def make_lonlat(vals=None, nrows=18, ncols=36):
    r = rast(nrows=nrows, ncols=ncols)
    if vals is None:
        vals = np.ones(nrows * ncols)
    return setValues(r, list(np.asarray(vals, dtype=float)))


# ── basic: lon/lat, all cells non-NA ─────────────────────────────────────────

def test_expanse_global_lonlat():
    r = make_lonlat()
    a = expanse(r)
    assert list(a.columns) == ["layer", "area"]
    assert len(a) == 1
    # earth surface area (geodesic) ~5.1e14 m2
    assert a["area"].iloc[0] == pytest.approx(5.1e14, rel=0.01)


def test_expanse_matches_cellsize_sum():
    r = make_lonlat()
    a = expanse(r)["area"].iloc[0]
    cs = pt.values(r.cellSize(mask=False)) if hasattr(r, "cellSize") else None
    if cs is None:
        from tappa.generics import cellSize
        cs = pt.values(cellSize(r, mask=False))
    assert a == pytest.approx(float(np.nansum(cs)), rel=1e-4)


def test_expanse_na_cells_excluded():
    vals = np.ones(18 * 36)
    vals[: 9 * 36] = np.nan  # northern hemisphere all NA
    r = make_lonlat(vals)
    full = expanse(make_lonlat())["area"].iloc[0]
    half = expanse(r)["area"].iloc[0]
    assert half == pytest.approx(full / 2, rel=1e-9)


def test_expanse_units():
    r = make_lonlat()
    m = expanse(r, unit="m")["area"].iloc[0]
    km = expanse(r, unit="km")["area"].iloc[0]
    ha = expanse(r, unit="ha")["area"].iloc[0]
    assert m / km == pytest.approx(1e6)
    assert m / ha == pytest.approx(1e4)


def test_expanse_invalid_unit():
    r = make_lonlat()
    with pytest.raises(RuntimeError):
        expanse(r, unit="miles")


# ── planar CRS, transform=False: exact xres*yres accounting ─────────────────

def test_expanse_planar_exact():
    r = rast(
        nrows=10, ncols=10,
        xmin=0, xmax=1000, ymin=0, ymax=1000,
        crs="EPSG:32633",
    )
    vals = np.ones(100)
    vals[:25] = np.nan
    r = setValues(r, list(vals))
    a = expanse(r, transform=False)
    assert a["area"].iloc[0] == pytest.approx(75 * 100 * 100)


# ── byValue ──────────────────────────────────────────────────────────────────

def test_expanse_by_value():
    vals = np.ones(18 * 36)
    vals[: 9 * 36] = 2.0
    r = make_lonlat(vals)
    a = expanse(r, byValue=True)
    assert list(a.columns) == ["layer", "value", "area"]
    assert sorted(a["value"].tolist()) == [1.0, 2.0]
    total = expanse(r)["area"].iloc[0]
    assert a["area"].sum() == pytest.approx(total, rel=1e-9)
    # symmetric split about the equator
    assert a["area"].iloc[0] == pytest.approx(a["area"].iloc[1], rel=1e-9)


def test_expanse_by_value_wide():
    vals = np.ones(18 * 36)
    vals[: 9 * 36] = 2.0
    r = make_lonlat(vals)
    w = expanse(r, byValue=True, wide=True)
    assert len(w) == 1
    assert 1.0 in w.columns and 2.0 in w.columns


# ── multi-layer ──────────────────────────────────────────────────────────────

def test_expanse_multilayer():
    r1 = make_lonlat()
    vals = np.ones(18 * 36)
    vals[: 9 * 36] = np.nan
    r2 = make_lonlat(vals)
    r = pt.rast([r1, r2]) if hasattr(pt, "rast") else None
    if r is None or not hasattr(r, "nlyr") or r.nlyr() != 2:
        from tappa.rast import rast as _rast
        r = _rast([r1, r2])
    a = expanse(r)
    assert a["layer"].tolist() == [0, 1]
    assert a["area"].iloc[1] == pytest.approx(a["area"].iloc[0] / 2, rel=1e-9)


def test_expanse_usenames():
    r = make_lonlat()
    r.names = ["abc"]
    a = expanse(r, usenames=True)
    assert a["layer"].tolist() == ["abc"]


# ── zones ────────────────────────────────────────────────────────────────────

def test_expanse_zones():
    r = make_lonlat()
    zvals = np.ones(18 * 36)
    zvals[: 9 * 36] = 2.0
    z = make_lonlat(zvals)
    a = expanse(r, zones=z)
    assert list(a.columns) == ["layer", "zone", "area"]
    assert len(a) == 2
    total = expanse(r)["area"].iloc[0]
    assert a["area"].sum() == pytest.approx(total, rel=1e-9)


def test_expanse_zones_by_value():
    vals = np.ones(18 * 36)
    vals[::2] = 3.0
    r = make_lonlat(vals)
    zvals = np.ones(18 * 36)
    zvals[: 9 * 36] = 2.0
    z = make_lonlat(zvals)
    a = expanse(r, zones=z, byValue=True)
    assert list(a.columns) == ["layer", "value", "zone", "area"]
    assert len(a) == 4  # 2 values x 2 zones
    total = expanse(r)["area"].iloc[0]
    assert a["area"].sum() == pytest.approx(total, rel=1e-9)


def test_expanse_zones_wide():
    r = make_lonlat()
    zvals = np.ones(18 * 36)
    zvals[: 9 * 36] = 2.0
    z = make_lonlat(zvals)
    w = expanse(r, zones=z, wide=True)
    assert len(w) == 1
    assert 1.0 in w.columns and 2.0 in w.columns


def test_expanse_zones_geom_mismatch():
    r = make_lonlat()
    z = rast(nrows=9, ncols=36)
    z = setValues(z, list(np.ones(9 * 36)))
    with pytest.raises(Exception):
        expanse(r, zones=z)


# ── method registration and vector dispatch unchanged ───────────────────────

def test_expanse_method_on_raster():
    r = make_lonlat()
    a = r.expanse()
    assert isinstance(a, pd.DataFrame)


def test_expanse_vector_still_works():
    from tappa.extent import ext
    from tappa.tessellate import tessellate
    h = tessellate(ext(0, 100, 0, 100), size=10)
    a = expanse(h, transform=False)
    assert isinstance(a, np.ndarray)
    assert (a > 0).all()
