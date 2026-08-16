"""Z coordinate storage and round-trips (terra #824)."""

import os
import tempfile

import numpy as np

import tappa as pt


def test_has_z_false_for_xy():
    v = pt.vect(np.column_stack([np.arange(1, 4), np.arange(10, 13)]))
    assert pt.has_z(v) is False


def test_setPointsXYZ_and_crds():
    v = pt.SpatVector()
    v.setPointsXYZ([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], [100.0, 200.0, 300.0])
    assert pt.has_z(v) is True
    xy = pt.crds(v)
    assert xy.shape == (3, 3)
    np.testing.assert_array_equal(xy[:, 0], [1, 2, 3])
    np.testing.assert_array_equal(xy[:, 2], [100, 200, 300])


def test_vect_xyz_matrix():
    v = pt.vect(np.column_stack([np.arange(1.0, 4), np.arange(10.0, 13), [100, 200, 300]]))
    assert pt.has_z(v) is True
    np.testing.assert_array_equal(pt.crds(v)[:, 2], [100, 200, 300])


def test_geom_vect_roundtrip():
    v = pt.vect(np.column_stack([np.arange(1.0, 4), np.arange(10.0, 13), [100, 200, 300]]))
    g = pt.geom(v)
    assert g.shape[1] == 6
    v2 = pt.vect(g, type="points")
    assert pt.has_z(v2) is True
    np.testing.assert_array_equal(pt.crds(v2)[:, 2], [100, 200, 300])


def test_write_roundtrip():
    from tappa.write import write

    v = pt.vect(np.column_stack([np.arange(1.0, 4), np.arange(10.0, 13), [100, 200, 300]]))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "z.gpkg")
        assert write(v, path, overwrite=True)
        v2 = pt.vect(path)
        assert pt.has_z(v2) is True
        np.testing.assert_array_equal(pt.crds(v2)[:, 2], [100, 200, 300])


def test_hex_includes_z():
    v = pt.vect(np.column_stack([np.arange(1.0, 4), np.arange(10.0, 13), [100, 200, 300]]))
    h = pt.geom(v, hex=True)[0].lower()
    # ISO Point Z (type 1001) or EWKB Point+Z — longer than XY-only
    xy_only = "0101000000000000000000f03f0000000000002440"
    assert len(h) > len(xy_only)
