"""
Tests for SpatRasterDataset (sds) and SprcCollection (sprc).

These mirror the usage patterns shown in the R terra package documentation
for sds() and sprc().  There are no equivalent tinytest files in the
inst/tinytest/ directory, so this is the canonical test suite for these
Python wrappers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa._terra import SpatOptions, SpatRaster
from tappa.rast import rast
from tappa.values import set_values, values as get_values
from tappa.sds import SpatRasterDataset, sds
from tappa.sprc import SprcCollection, sprc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rast(vals, nrow=3, ncol=3) -> SpatRaster:
    r = rast(nrows=nrow, ncols=ncol, xmin=0, xmax=1, ymin=0, ymax=1)
    return set_values(r, list(float(v) for v in vals))


def _vals(r) -> np.ndarray:
    r.readStart()
    try:
        v = np.array(r.readValues(0, r.nrow(), 0, r.ncol()), dtype=float)
    finally:
        r.readStop()
    return v


# ---------------------------------------------------------------------------
# sds — SpatRasterDataset
# ---------------------------------------------------------------------------

class TestSds:

    def test_sds_empty(self):
        ds = sds()
        assert len(ds) == 0
        assert isinstance(ds, SpatRasterDataset)

    def test_sds_from_list(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        assert len(ds) == 2
        assert isinstance(ds, SpatRasterDataset)

    def test_sds_from_spatrasters_positional(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        r3 = _make_rast(range(21, 30))
        ds = sds(r1, r2, r3)
        assert len(ds) == 3

    def test_sds_getitem_int(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        got = ds[0]
        assert isinstance(got, SpatRaster)
        np.testing.assert_array_almost_equal(_vals(got), np.arange(1, 10, dtype=float))

    def test_sds_getitem_negative(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        got = ds[-1]
        np.testing.assert_array_almost_equal(_vals(got), np.arange(11, 20, dtype=float))

    def test_sds_getitem_out_of_range(self):
        ds = sds([_make_rast(range(9))])
        with pytest.raises(IndexError):
            _ = ds[5]

    def test_sds_iter(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        items = list(ds)
        assert len(items) == 2
        assert all(isinstance(r, SpatRaster) for r in items)

    def test_sds_names_default(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        nms = ds.names
        assert isinstance(nms, list)
        assert len(nms) == 2

    def test_sds_names_set(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        ds.names = ["temperature", "precipitation"]
        assert ds.names == ["temperature", "precipitation"]

    def test_sds_getitem_by_name(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        ds.names = ["temp", "prec"]
        got = ds["prec"]
        np.testing.assert_array_almost_equal(_vals(got), np.arange(11, 20, dtype=float))

    def test_sds_getitem_unknown_name(self):
        ds = sds([_make_rast(range(9))])
        with pytest.raises(KeyError):
            _ = ds["nonexistent"]

    def test_sds_nsds(self):
        ds = sds([_make_rast(range(9)), _make_rast(range(9))])
        assert ds.nsds() == 2

    def test_sds_nrow_ncol(self):
        r = _make_rast(range(9), nrow=3, ncol=3)
        ds = sds([r])
        assert ds.nrow() == 3
        assert ds.ncol() == 3

    def test_sds_nlyr(self):
        r1 = _make_rast(range(9))
        r2 = _make_rast(range(9))
        ds = sds([r1, r2])
        lyrs = ds.nlyr()
        assert isinstance(lyrs, list)
        assert all(v == 1 for v in lyrs)

    def test_sds_collapse(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        ds = sds([r1, r2])
        coll = ds.collapse()
        assert isinstance(coll, SpatRaster)
        assert int(coll.nlyr()) == 2
        v = _vals(coll)
        assert len(v) == 18  # 2 layers × 9 cells

    def test_sds_subset(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        r3 = _make_rast(range(21, 30))
        ds = sds([r1, r2, r3])
        sub = ds.subset([0, 2])
        assert isinstance(sub, SpatRasterDataset)
        assert len(sub) == 2

    def test_sds_repr(self):
        r1 = _make_rast(range(9))
        ds = sds([r1])
        s = repr(ds)
        assert "SpatRasterDataset" in s
        assert "1 sub-dataset" in s

    def test_sds_second_dataset_values(self):
        """Values in each sub-dataset must remain independent."""
        r1 = _make_rast([float(i) for i in range(9)])
        r2 = _make_rast([float(i) * 2 for i in range(9)])
        ds = sds([r1, r2])
        v1 = _vals(ds[0])
        v2 = _vals(ds[1])
        np.testing.assert_array_almost_equal(v2, v1 * 2)


# ---------------------------------------------------------------------------
# sprc — SprcCollection
# ---------------------------------------------------------------------------

class TestSprc:

    def test_sprc_empty(self):
        rc = sprc()
        assert len(rc) == 0
        assert isinstance(rc, SprcCollection)

    def test_sprc_from_list(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        rc = sprc([r1, r2])
        assert len(rc) == 2
        assert isinstance(rc, SprcCollection)

    def test_sprc_from_positional(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        r3 = _make_rast(range(21, 30))
        rc = sprc(r1, r2, r3)
        assert len(rc) == 3

    def test_sprc_getitem_int(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        rc = sprc([r1, r2])
        got = rc[0]
        assert isinstance(got, SpatRaster)
        np.testing.assert_array_almost_equal(_vals(got), np.arange(1, 10, dtype=float))

    def test_sprc_getitem_negative(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        rc = sprc([r1, r2])
        got = rc[-1]
        np.testing.assert_array_almost_equal(_vals(got), np.arange(11, 20, dtype=float))

    def test_sprc_getitem_out_of_range(self):
        rc = sprc([_make_rast(range(9))])
        with pytest.raises(IndexError):
            _ = rc[10]

    def test_sprc_iter(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        rc = sprc([r1, r2])
        items = list(rc)
        assert len(items) == 2
        assert all(isinstance(r, SpatRaster) for r in items)

    def test_sprc_names(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        rc = sprc([r1, r2])
        rc.names = ["a", "b"]
        assert rc.names == ["a", "b"]

    def test_sprc_getitem_by_name(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        rc = sprc([r1, r2])
        rc.names = ["alpha", "beta"]
        got = rc["beta"]
        np.testing.assert_array_almost_equal(_vals(got), np.arange(11, 20, dtype=float))

    def test_sprc_length(self):
        rc = sprc([_make_rast(range(9)), _make_rast(range(9))])
        assert rc.length() == 2

    def test_sprc_add(self):
        rc = sprc()
        r = _make_rast(range(9))
        rc.add(r)
        assert len(rc) == 1

    def test_sprc_merge_non_overlapping(self):
        """Merge two non-overlapping rasters; each tile should have its values."""
        # tile 1: x=[0,1], tile 2: x=[1,2], same y=[0,1]
        opt = SpatOptions()
        r1 = rast(nrows=3, ncols=3, xmin=0, xmax=1, ymin=0, ymax=1)
        r1 = set_values(r1, [float(v) for v in range(1, 10)])
        r2 = rast(nrows=3, ncols=3, xmin=1, xmax=2, ymin=0, ymax=1)
        r2 = set_values(r2, [float(v) * 10 for v in range(1, 10)])
        rc = sprc([r1, r2])
        merged = rc.merge()
        assert isinstance(merged, SpatRaster)
        # merged raster should have 3 rows and 6 columns
        assert int(merged.ncol()) == 6
        assert int(merged.nrow()) == 3

    def test_sprc_mosaic_overlapping(self):
        """Mosaic with 'mean': overlapping cells get the mean of their values."""
        # two identical rasters – mean of (v, v) == v
        r1 = _make_rast([float(v) for v in range(1, 10)])
        r2 = _make_rast([float(v) for v in range(1, 10)])
        rc = sprc([r1, r2])
        mos = rc.mosaic(fun="mean")
        assert isinstance(mos, SpatRaster)
        result = _vals(mos)
        expected = np.arange(1, 10, dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sprc_mosaic_mean_overlap(self):
        """Cells covered by both rasters are averaged in mosaic."""
        # Both rasters on the same grid; r2 has doubled values.
        # mosaic(mean) → mean(v, 2v) = 1.5v
        r1 = _make_rast([float(v) for v in range(2, 11)])
        r2 = _make_rast([float(v) * 2 for v in range(2, 11)])
        rc = sprc([r1, r2])
        mos = rc.mosaic(fun="mean")
        result = _vals(mos)
        expected = np.arange(2, 11, dtype=float) * 1.5
        np.testing.assert_array_almost_equal(result, expected)

    def test_sprc_repr(self):
        r1 = _make_rast(range(9))
        rc = sprc([r1])
        s = repr(rc)
        assert "SprcCollection" in s
        assert "1 raster" in s

    def test_sprc_getitem_slice(self):
        r1 = _make_rast(range(1, 10))
        r2 = _make_rast(range(11, 20))
        r3 = _make_rast(range(21, 30))
        rc = sprc([r1, r2, r3])
        sub = rc[1:]
        assert isinstance(sub, SprcCollection)
        assert len(sub) == 2

    def test_sds_and_sprc_round_trip(self):
        """Values should be faithfully preserved through sds / sprc."""
        data = [float(i) * 1.5 for i in range(9)]
        r_orig = _make_rast(data)

        ds = sds([r_orig])
        r_back_sds = ds[0]
        np.testing.assert_array_almost_equal(_vals(r_back_sds), np.array(data))

        rc = sprc([r_orig])
        r_back_sprc = rc[0]
        np.testing.assert_array_almost_equal(_vals(r_back_sprc), np.array(data))
