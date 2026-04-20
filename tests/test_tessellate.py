"""
Tests for tappa.tessellate (parallel to R inst/tinytest/test_tessellate.R).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa.extent import ext
from tappa.spatvec import expanse, crds
from tappa.tessellate import tessellate


def _attr(v, name):
    """Return attribute column ``name`` of SpatVector *v* as a list."""
    df = v.df
    if hasattr(df, "values") and callable(getattr(df, "values")):
        d = df.values()
    else:
        d = df
    if name in d:
        return list(d[name])
    return []


# ── planar hex grid: exact equal-area tessellation ──────────────────────────

def test_planar_hex_equal_area_pointy():
    e = ext(0, 100, 0, 100)
    h = tessellate(e, size=10)
    assert h.nrow() > 0
    a = expanse(h, transform=False)
    assert (a.max() - a.min()) / a.mean() < 1e-9


def test_planar_hex_equal_area_flat_top():
    e = ext(0, 100, 0, 100)
    h = tessellate(e, size=12, flat_top=True)
    assert h.nrow() > 0
    a = expanse(h, transform=False)
    assert (a.max() - a.min()) / a.mean() < 1e-9


# ── global geo=True equal-area on the sphere ────────────────────────────────

def test_global_geo_equal_area_count():
    g = tessellate(ext(-180, 180, -85, 85), size=500000, geo=True)
    assert g.nrow() == 2320
    ag = expanse(g)
    assert (ag.max() - ag.min()) / ag.mean() < 0.02


# ── adjacent regional grids share the global anchor ─────────────────────────

def test_geo_adjacent_grids_share_anchor():
    g_eu = tessellate(ext(-10, 40, 30, 70), size=200000, geo=True)
    g_af = tessellate(ext(10, 60, -5, 35), size=200000, geo=True)
    cn_eu = g_eu.centroid(False).coordinates()
    cn_af = g_af.centroid(False).coordinates()
    cn_eu = np.column_stack([cn_eu[0], cn_eu[1]])
    cn_af = np.column_stack([cn_af[0], cn_af[1]])

    def in_overlap(cn):
        return ((cn[:, 0] >= 10.5) & (cn[:, 0] <= 39.5)
                & (cn[:, 1] >= 30.5) & (cn[:, 1] <= 34.5))

    sub_eu = cn_eu[in_overlap(cn_eu)]
    sub_af = cn_af[in_overlap(cn_af)]
    assert sub_eu.shape[0] == sub_af.shape[0]
    oe = np.lexsort((sub_eu[:, 1], sub_eu[:, 0]))
    oa = np.lexsort((sub_af[:, 1], sub_af[:, 0]))
    np.testing.assert_allclose(sub_eu[oe], sub_af[oa])


# ── antimeridian cells: equal counts per latitude row ───────────────────────

def test_antimeridian_equal_per_row():
    h180 = tessellate(ext(-180, 180, -80, 80), size=500000, geo=True)
    cn = h180.centroid(False).coordinates()
    lats = np.round(np.array(cn[1]), 4)
    _, counts = np.unique(lats, return_counts=True)
    assert np.all(counts == counts[0])


# ── input validation ────────────────────────────────────────────────────────

def test_size_validation():
    e = ext(0, 100, 0, 100)
    with pytest.raises((ValueError, RuntimeError)):
        tessellate(e, size=0)
    with pytest.raises((ValueError, RuntimeError)):
        tessellate(e, size=-1)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        tessellate(e, size=[1, 2])


def test_polyhedron_n_validation():
    with pytest.raises((ValueError, RuntimeError)):
        tessellate(n=0, type="polyhedron")
    with pytest.raises((ValueError, RuntimeError)):
        tessellate(size=0, type="polyh")


# ── polyhedron base case (Goldberg G(2, 0)) ─────────────────────────────────

def test_polyhedron_n2():
    g = tessellate(n=2, type="polyhedrons")
    assert g.nrow() == 42
    types = _attr(g, "type")
    assert sum(t == "pentagon" for t in types) == 12


# ── n = 10 -> 1002 cells ────────────────────────────────────────────────────

def test_polyhedron_n10():
    g10 = tessellate(n=10, type="polyhedron")
    assert g10.nrow() == 1002
    types = _attr(g10, "type")
    assert sum(t == "pentagon" for t in types) == 12
    assert sum(t == "hexagon" for t in types) == 990
    assert all(g10.geos_isvalid())


# ── total area is invariant across resolutions ──────────────────────────────

def test_polyhedron_constant_total_area():
    ref_area = expanse(tessellate(n=2, type="polyhedron")).sum()
    for n in (2, 5, 10, 20):
        g = tessellate(n=n, type="polyhedron")
        assert g.nrow() == 10 * n * n + 2
        assert all(g.geos_isvalid())
        a = expanse(g).sum()
        assert abs(a - ref_area) / ref_area < 1e-6


# ── two pentagons sit on the geographic poles ───────────────────────────────

def test_polyhedron_polar_pentagons():
    g = tessellate(n=5, type="polyhedron")
    types = _attr(g, "type")
    pent_idx = [i for i, t in enumerate(types) if t == "pentagon"]
    pent = g.subset_rows(pent_idx)
    ymax = []
    ymin = []
    for i in range(pent.nrow()):
        e = pent.subset_rows([i]).extent()
        v = list(e.vector)
        ymin.append(v[2])
        ymax.append(v[3])
    assert any(y >= 90 - 1e-9 for y in ymax)
    assert any(y <= -90 + 1e-9 for y in ymin)


# ── size-derived n ──────────────────────────────────────────────────────────

def test_polyhedron_size_argument():
    g_size = tessellate(size=2e6, type="polyhedron")
    assert g_size.nrow() > 0
    types = set(_attr(g_size, "type"))
    assert types <= {"pentagon", "hexagon"}
    g_small = tessellate(size=5e5, type="polyhedron")
    assert g_small.nrow() > g_size.nrow()


# ── extent filter selects a subset including polar edges ────────────────────

def test_polyhedron_extent_filter():
    g10 = tessellate(n=10, type="polyhedron")
    g_eq = tessellate(ext(-30, 30, -10, 10), n=10, type="polyhedron", geo=True)
    assert g_eq.nrow() > 0
    assert g_eq.nrow() < g10.nrow()
