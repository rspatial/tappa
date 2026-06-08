"""
Tests for the SpatVector overlay helpers (points / lines / polys) and for
the multi-SpatRaster form of pt.rast().
"""
from __future__ import annotations

import numpy as np
import pytest
import tappa as pt

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Stacking SpatRasters via pt.rast([r1, r2, r3])
# ---------------------------------------------------------------------------

def test_rast_stack_from_spatraster_list():
    r1 = pt.set_values(pt.rast(nrows=4, ncols=5),
                       np.arange(1, 21, dtype=float))
    r2 = pt.set_values(pt.rast(nrows=4, ncols=5),
                       np.arange(21, 41, dtype=float))
    r3 = pt.set_values(pt.rast(nrows=4, ncols=5),
                       np.arange(41, 61, dtype=float))

    s = pt.rast([r1, r2, r3])
    assert pt.nlyr(s) == 3
    assert pt.nrow(s) == pt.nrow(r1)
    assert pt.ncol(s) == pt.ncol(r1)

    vals = pt.values(s)
    np.testing.assert_array_equal(vals[:, 0], np.arange(1, 21))
    np.testing.assert_array_equal(vals[:, 1], np.arange(21, 41))
    np.testing.assert_array_equal(vals[:, 2], np.arange(41, 61))


def test_rast_stack_single_element_returns_deep_copy():
    r = pt.set_values(pt.rast(nrows=2, ncols=2),
                      np.array([1.0, 2.0, 3.0, 4.0]))
    s = pt.rast([r])
    assert pt.nlyr(s) == 1
    np.testing.assert_array_equal(
        pt.values(s).ravel(), pt.values(r).ravel()
    )


# ---------------------------------------------------------------------------
# points / lines / polys
# ---------------------------------------------------------------------------

def _make_points():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
    return pt.vect(xy, crs="OGC:CRS84")


def _make_polygon():
    return pt.vect("POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))", crs="OGC:CRS84")


def _make_line():
    return pt.vect("LINESTRING (0 0, 1 1, 2 0.5)", crs="OGC:CRS84")


def test_points_draws_on_existing_axes():
    fig, ax = plt.subplots()
    out = pt.points(_make_points(), ax=ax, col="red", cex=2)
    assert out is ax
    # one PathCollection per scatter call
    assert any(c.__class__.__name__ == "PathCollection" for c in ax.collections)


def test_points_rejects_non_point_input():
    with pytest.raises(ValueError, match="expected 'points'"):
        pt.points(_make_polygon())


def test_lines_handles_polygon_by_closing_ring():
    fig, ax = plt.subplots()
    pt.lines(_make_polygon(), ax=ax, col="blue", lwd=2)
    # Should produce at least one Line2D with closed coordinates
    line2ds = [ln for ln in ax.lines]
    assert len(line2ds) == 1
    xs, ys = line2ds[0].get_xdata(), line2ds[0].get_ydata()
    assert (xs[0], ys[0]) == (xs[-1], ys[-1])
    assert line2ds[0].get_color() == "blue"
    assert line2ds[0].get_linewidth() == 2


def test_lines_converts_points_to_connected_segments():
    fig, ax = plt.subplots()
    pt.lines(_make_points(), ax=ax, col="red", lwd=2)
    assert len(ax.lines) == 1
    xs, ys = ax.lines[0].get_data()
    assert len(xs) == 3
    assert len(ys) == 3


def test_polys_filled_default_outline_only():
    fig, ax = plt.subplots()
    pt.polys(_make_polygon(), ax=ax, col="yellow", border="black", lwd=3)
    patches = ax.patches
    assert len(patches) == 1
    p = patches[0]
    # Polygon vertex count: 4 vertices, matplotlib auto-closes.
    assert len(p.get_xy()) >= 4
    assert p.get_edgecolor()[3] == 1.0  # solid edge alpha
    assert p.get_linewidth() == 3


def test_polys_rejects_non_polygon():
    with pytest.raises(ValueError, match="expected 'polygons'"):
        pt.polys(_make_line())


def test_overlays_compose_on_a_raster_axes():
    """`pt.plot(r)` -> `pt.points(...)` should land on the same Axes."""
    r = pt.set_values(pt.rast(nrows=4, ncols=4, xmin=0, xmax=4, ymin=0, ymax=4),
                      np.arange(1, 17, dtype=float))
    ax = pt.plot(r, figsize=(3, 3))
    pts = _make_points()
    pt.points(pts, ax=ax, col="red", cex=2)
    # Existing image artists for raster + at least one scatter on the same Axes.
    assert any(im.__class__.__name__ == "AxesImage" for im in ax.images)
    assert any(c.__class__.__name__ == "PathCollection" for c in ax.collections)
