"""Tests for rast(matrix), ncol/nrow aliases, and as_points on lines."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa.rast import rast


def test_rast_from_matrix():
    mi = np.arange(1, 37, dtype=float).reshape(6, 6)
    ri = rast(mi)
    assert ri.nrow() == 6
    assert ri.ncol() == 6
    assert ri.nlyr() == 1
    assert ri.xmin() == 0.0
    assert ri.xmax() == 6.0
    assert ri.ymin() == 0.0
    assert ri.ymax() == 6.0
    got = np.array(ri.readValues(0, 6, 0, 6), dtype=float).reshape(6, 6, order="F")
    np.testing.assert_array_equal(got, mi)


def test_rast_resolution():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    geom = np.column_stack([
        np.ones(len(pts), dtype=int),
        np.ones(len(pts), dtype=int),
        pts[:, 0],
        pts[:, 1],
    ])
    poly = pt.vect(geom, type="polygons", crs="local")
    r = rast(poly, resolution=2)
    assert r.nrow() == 5
    assert r.ncol() == 5
    assert pt.res(r) == [2.0, 2.0]


def test_rast_ncol_nrow_aliases():
    r = rast(xmin=0.5, xmax=1.4, ymin=0.6, ymax=1.5, ncol=9, nrow=9, crs="")
    assert r.nrow() == 9
    assert r.ncol() == 9
    assert r.ncell() == 81


def test_as_points_on_lines():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    geom = np.column_stack([
        np.ones(len(pts), dtype=int),
        np.ones(len(pts), dtype=int),
        pts[:, 0],
        pts[:, 1],
    ])
    lines = pt.vect(geom, type="lines", crs="local")
    g = pt.as_points(lines)
    assert g.type() == "points"
    assert g.nrow() == 3
    xy = pt.geom(g)[:, 2:4]
    np.testing.assert_allclose(xy, pts)
