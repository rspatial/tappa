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


# ── default md, single subds (t2m) ───────────────────────────────────────────

class TestDefaultSingleSubds:
    """rast(f, subds='t2m') — default md (auto-detect, md=None → C++ 2)."""

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


# ── md=True vs default equivalence (single subds) ────────────────────────────

class TestMdEquivalence:
    """For a single subds, default and md=True must give identical results."""

    def test_values_equal(self, nc_path):
        r1 = rast(nc_path, subds="t2m")
        r2 = rast(nc_path, subds="t2m", md=True)
        np.testing.assert_array_equal(values(r1), values(r2))

    def test_extract_equal(self, nc_path, xy_point):
        r1 = rast(nc_path, subds="t2m")
        r2 = rast(nc_path, subds="t2m", md=True)
        e1 = extract(r1, xy_point)
        e2 = extract(r2, xy_point)
        np.testing.assert_array_equal(
            e1.select_dtypes(include="number").values,
            e2.select_dtypes(include="number").values,
        )


# ── default md, all vars (no subds) ──────────────────────────────────────────

class TestDefaultAllVars:
    """rast(f) — all variables, default md."""

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


# ── md=True, all vars — equivalence with default ────────────────────────────

class TestMdTrueAllVars:
    """md=True must give identical results to default for all vars."""

    def test_nlyr(self, nc_path):
        r = rast(nc_path, md=True)
        assert r.nlyr() == 336

    def test_values_match_default(self, nc_path):
        r1 = rast(nc_path)
        r2 = rast(nc_path, md=True)
        np.testing.assert_array_equal(values(r1), values(r2))

    def test_extract_match_default(self, nc_path, xy_point):
        r1 = rast(nc_path)
        r2 = rast(nc_path, md=True)
        e1 = extract(r1, xy_point)
        e2 = extract(r2, xy_point)
        np.testing.assert_array_equal(
            e1.select_dtypes(include="number").values,
            e2.select_dtypes(include="number").values,
        )


# ── md=False vs md=True: same values after aligning by layer name ────────────

def _align_values_by_name(r_from, r_to):
    """Subset and reorder values of *r_from* to match layer names of *r_to*.

    Drops layers from *r_from* whose names are not in *r_to* (e.g.
    NaN-filled ``expver`` layers that md=False introduces).
    """
    names_from = list(r_from.names)
    names_to = list(r_to.names)
    idx = [names_from.index(n) for n in names_to]
    return values(r_from)[:, idx]


class TestMdFalseVsTrueByName:
    """md=False may add extra NaN layers, but values for shared layer names
    must be identical to md=True."""

    def test_single_subds_values(self, nc_path):
        r0 = rast(nc_path, subds="t2m", md=False)
        r1 = rast(nc_path, subds="t2m", md=True)
        v0_aligned = _align_values_by_name(r0, r1)
        np.testing.assert_allclose(v0_aligned, values(r1), atol=1e-6)

    def test_single_subds_extract(self, nc_path, xy_point):
        r0 = rast(nc_path, subds="t2m", md=False)
        r1 = rast(nc_path, subds="t2m", md=True)
        e0 = extract(r0, xy_point)
        e1 = extract(r1, xy_point)
        np.testing.assert_allclose(
            e0[list(e1.columns)].values.astype(float),
            e1.values.astype(float),
            atol=1e-6,
        )

    def test_all_vars_values(self, nc_path):
        r0 = rast(nc_path, md=False)
        r1 = rast(nc_path, md=True)
        v0_aligned = _align_values_by_name(r0, r1)
        np.testing.assert_allclose(v0_aligned, values(r1), atol=1e-6)

    def test_all_vars_extract(self, nc_path, xy_point):
        r0 = rast(nc_path, md=False)
        r1 = rast(nc_path, md=True)
        e0 = extract(r0, xy_point)
        e1 = extract(r1, xy_point)
        np.testing.assert_allclose(
            e0[list(e1.columns)].values.astype(float),
            e1.values.astype(float),
            atol=1e-6,
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
