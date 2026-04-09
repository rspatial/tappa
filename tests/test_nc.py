"""
Tests for opening NetCDF files with md=True/False, subds, and sds().

Mirrors the R tinytest tests in inst/tinytest/test_nc.R using
the same reference values from nouragues.nc.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa.rast import rast
from tappa.values import values
from tappa.extract import extract
from tappa.names import varnames
from tappa.vect import vect
from tappa.sds import sds

from path_utils import skip_if_missing_inst_ex


@pytest.fixture
def nc_path():
    return skip_if_missing_inst_ex("nouragues.nc")


@pytest.fixture
def xy_point():
    return np.array([[-52.67, 4.08]])


# ── md=False (default), single subds (t2m) ───────────────────────────────────

class TestMdFalseSingleSubds:

    def test_nlyr(self, nc_path):
        r = rast(nc_path, subds="t2m")
        assert r.nlyr() == 48

    def test_dim(self, nc_path):
        r = rast(nc_path, subds="t2m")
        assert (r.nrow(), r.ncol(), r.nlyr()) == (3, 3, 48)

    def test_values_cell_1_1(self, nc_path):
        r = rast(nc_path, subds="t2m")
        v = values(r)
        assert v.shape == (9, 48)
        np.testing.assert_allclose(v[0, 0], 296.9571, atol=1e-3)

    def test_values_cell_9_1(self, nc_path):
        r = rast(nc_path, subds="t2m")
        v = values(r)
        np.testing.assert_allclose(v[8, 0], 297.0769, atol=1e-3)

    def test_values_cell_1_48(self, nc_path):
        r = rast(nc_path, subds="t2m")
        v = values(r)
        np.testing.assert_allclose(v[0, 47], 296.9695, atol=1e-3)

    def test_extract(self, nc_path, xy_point):
        r = rast(nc_path, subds="t2m")
        e = extract(r, xy_point)
        val = e.iloc[0, 1]
        np.testing.assert_allclose(float(val), 297.0172, atol=1e-3)


# ── md=True, single subds (t2m) ──────────────────────────────────────────────

class TestMdTrueSingleSubds:

    def test_nlyr(self, nc_path):
        r = rast(nc_path, subds="t2m", md=True)
        assert r.nlyr() == 48

    def test_dim(self, nc_path):
        r = rast(nc_path, subds="t2m", md=True)
        assert (r.nrow(), r.ncol(), r.nlyr()) == (3, 3, 48)

    def test_values_cell_1_1(self, nc_path):
        r = rast(nc_path, subds="t2m", md=True)
        v = values(r)
        np.testing.assert_allclose(v[0, 0], 296.9571, atol=1e-3)

    def test_values_cell_9_1(self, nc_path):
        r = rast(nc_path, subds="t2m", md=True)
        v = values(r)
        np.testing.assert_allclose(v[8, 0], 297.0769, atol=1e-3)

    def test_extract(self, nc_path, xy_point):
        r = rast(nc_path, subds="t2m", md=True)
        e = extract(r, xy_point)
        val = e.iloc[0, 1]
        np.testing.assert_allclose(float(val), 297.0172, atol=1e-3)


# ── md=True vs md=False equivalence ──────────────────────────────────────────

class TestMdEquivalence:

    def test_values_equal(self, nc_path):
        r1 = rast(nc_path, subds="t2m", md=False)
        r2 = rast(nc_path, subds="t2m", md=True)
        np.testing.assert_array_equal(values(r1), values(r2))

    def test_extract_equal(self, nc_path, xy_point):
        r1 = rast(nc_path, subds="t2m", md=False)
        r2 = rast(nc_path, subds="t2m", md=True)
        e1 = extract(r1, xy_point)
        e2 = extract(r2, xy_point)
        np.testing.assert_array_equal(
            e1.select_dtypes(include="number").values,
            e2.select_dtypes(include="number").values,
        )


# ── md=False, all vars (no subds) ────────────────────────────────────────────

class TestMdFalseAllVars:

    def test_nlyr(self, nc_path):
        r = rast(nc_path)
        assert r.nlyr() == 336

    def test_dim(self, nc_path):
        r = rast(nc_path)
        assert (r.nrow(), r.ncol(), r.nlyr()) == (3, 3, 336)

    def test_varnames(self, nc_path):
        r = rast(nc_path)
        vn = varnames(r)
        for name in ("d2m", "sp", "ssrd", "t2m", "tp", "u10", "v10"):
            assert name in vn

    def test_first_value(self, nc_path):
        r = rast(nc_path)
        v = values(r)
        np.testing.assert_allclose(v[0, 0], 295.4049, atol=1e-3)

    def test_extract_ncol(self, nc_path, xy_point):
        r = rast(nc_path)
        e = extract(r, xy_point)
        assert e.shape[1] == 336 or e.shape[1] == 337  # 336 + optional ID


# ── md=True, all vars ────────────────────────────────────────────────────────

class TestMdTrueAllVars:

    def test_nlyr(self, nc_path):
        r = rast(nc_path, md=True)
        assert r.nlyr() == 336

    def test_values_match_md_false(self, nc_path):
        r1 = rast(nc_path, md=False)
        r2 = rast(nc_path, md=True)
        np.testing.assert_array_equal(values(r1), values(r2))

    def test_extract_match_md_false(self, nc_path, xy_point):
        r1 = rast(nc_path, md=False)
        r2 = rast(nc_path, md=True)
        e1 = extract(r1, xy_point)
        e2 = extract(r2, xy_point)
        np.testing.assert_array_equal(
            e1.select_dtypes(include="number").values,
            e2.select_dtypes(include="number").values,
        )


# ── SpatRasterDataset ────────────────────────────────────────────────────────

class TestSds:

    def test_nsds(self, nc_path):
        d = sds(nc_path)
        assert len(d) == 7

    def test_names(self, nc_path):
        d = sds(nc_path)
        for name in ("u10", "v10", "d2m", "t2m", "sp", "tp", "ssrd"):
            assert name in d.names

    def test_u10_nlyr(self, nc_path):
        d = sds(nc_path)
        du = d["u10"]
        assert du.nlyr() == 48

    def test_u10_dim(self, nc_path):
        d = sds(nc_path)
        du = d["u10"]
        assert (du.nrow(), du.ncol(), du.nlyr()) == (3, 3, 48)

    def test_u10_values(self, nc_path):
        d = sds(nc_path)
        du = d["u10"]
        v = values(du)
        np.testing.assert_allclose(v[0, 0], -1.2086, atol=1e-3)
        np.testing.assert_allclose(v[8, 47], -1.0656, atol=1e-3)

    def test_u10_extract(self, nc_path, xy_point):
        d = sds(nc_path)
        du = d["u10"]
        e = extract(du, xy_point)
        val = e.iloc[0, 1]
        np.testing.assert_allclose(float(val), -1.1612, atol=1e-3)
