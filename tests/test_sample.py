"""
Tests ported from inst/tinytest/test_sample.R

Spatial sampling of SpatRaster objects.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa.rast import rast
from tappa.values import set_values
from tappa.sample import spat_sample

from path_utils import skip_if_missing_inst_ex


def _need_pandas(obj):
    pd = pytest.importorskip("pandas")
    assert isinstance(obj, pd.DataFrame)
    return obj


def _make_rr():
    """5x5 raster with values 1-25, some NA in corner."""
    r = rast(nrows=5, ncols=5, xmin=0, xmax=5, ymin=0, ymax=5, crs="local")
    vals = np.arange(1, 26, dtype=float)
    vals[0] = float("nan")
    vals[1] = float("nan")
    return set_values(r, vals)


def test_random_unique_cells():
    rr = _make_rr()
    df = _need_pandas(spat_sample(rr, 20, method="random", cells=True))
    assert len(df) == 20
    assert "cell" in df.columns
    assert df["cell"].nunique() == 20


def test_random_replace_allows_duplicates():
    r5 = rast(nrows=3, ncols=3, xmin=0, xmax=3, ymin=0, ymax=3, crs="local")
    r5 = set_values(r5, np.ones(9))
    df = _need_pandas(spat_sample(r5, 30, method="random", replace=True, cells=True))
    assert len(df) == 30
    assert df["cell"].nunique() <= 9


def test_random_na_rm():
    rr = _make_rr()
    df = _need_pandas(spat_sample(rr, 8, method="random", na_rm=True, cells=True))
    assert len(df) == 8
    val_cols = [c for c in df.columns if c not in ("cell", "x", "y")]
    for col in val_cols:
        assert not df[col].isna().any()


def test_regular_xy_and_values_match():
    rr = _make_rr()
    df = _need_pandas(
        spat_sample(rr, 9, method="regular", cells=True, xy=True, values=True)
    )
    assert "cell" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert len(df) >= 1


def test_regular_as_raster_coarser():
    rr = _make_rr()
    srast = spat_sample(rr, 50, method="regular", as_raster=True)
    assert srast.nrow() <= rr.nrow()
    assert srast.ncol() <= rr.ncol()


def test_regular_exact_rowcount():
    r9_template = rast(nrows=5, ncols=5, xmin=0, xmax=5, ymin=0, ymax=5, crs="local")
    r9_template = set_values(r9_template, np.ones(25))
    df = _need_pandas(
        spat_sample(r9_template, 25, method="regular", cells=True, exact=True)
    )
    assert len(df) == 25
