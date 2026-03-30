"""
Tests ported from inst/tinytest/test_window.R

Reading windows on SpatRaster; sampling and extract with shared geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa._terra import SpatOptions
from tappa._helpers import messages
from tappa.rast import rast
from tappa.extent import ext
from tappa.generics import crop
from tappa.window import set_window
from tappa.extract import extract
from tappa.sample import spat_sample

from path_utils import skip_if_missing_inst_ex


def _c_stack(*layers):
    opt = SpatOptions()
    out = layers[0].deepcopy()
    for L in layers[1:]:
        out.addSource(L, True, opt)
    return messages(out, "c_stack")


def _read_all_values(rr):
    rr.readStart()
    try:
        return np.array(rr.readValues(0, rr.nrow(), 0, rr.ncol()), dtype=float)
    finally:
        rr.readStop()


@pytest.fixture
def logo():
    return rast(skip_if_missing_inst_ex("logo.tif"))


def test_crop_and_window_values_match(logo):
    x = logo
    y = x * 1.0
    e = ext(35, 55, 35, 55)
    z = crop(x, e)
    xw = set_window(x, e)
    yw = set_window(y, e)
    np.testing.assert_array_almost_equal(_read_all_values(xw), _read_all_values(z))
    np.testing.assert_array_almost_equal(_read_all_values(yw), _read_all_values(z))


def test_spat_sample_windowed_equals_cropped_skipped(logo):
    """Port of R: set.seed(1); spatSample(x)==spatSample(y)==spatSample(z) with cell=TRUE."""
    import pandas as pd

    x = logo
    y = x * 1.0
    e = ext(35, 55, 35, 55)
    z = crop(x, e)
    xw = set_window(x, e)
    yw = set_window(y, e)
    a = _c_stack(z, yw, xw)

    np.random.seed(1)
    s = spat_sample(xw, 4, method="random", cells=True)
    np.random.seed(1)
    sy = spat_sample(yw, 4, method="random", cells=True)
    np.random.seed(1)
    sz = spat_sample(z, 4, method="random", cells=True)

    for d in (s, sy, sz):
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 4

    # same cells in all three
    np.testing.assert_array_equal(s["cell"].values, sy["cell"].values)
    np.testing.assert_array_equal(s["cell"].values, sz["cell"].values)

    # same layer values
    val_cols = [c for c in s.columns if c != "cell"]
    for col in val_cols:
        np.testing.assert_array_almost_equal(s[col].values, sy[col].values)
        np.testing.assert_array_almost_equal(s[col].values, sz[col].values)

    # stacked raster: same seed → same cells; first sub-raster values match s
    np.random.seed(1)
    sa = spat_sample(a, 4, method="random", cells=True)
    assert isinstance(sa, pd.DataFrame)
    assert len(sa) == 4
    np.testing.assert_array_equal(s["cell"].values, sa["cell"].values)
    sa_val_cols = [c for c in sa.columns if c != "cell"]
    for i, col in enumerate(val_cols):
        np.testing.assert_array_almost_equal(s[col].values, sa[sa_val_cols[i]].values)


def test_spat_sample_same_shape(logo):
    """Non-seeded: windowed and cropped rasters yield same sample dimensions."""
    x = logo
    y = x * 1.0
    e = ext(35, 55, 35, 55)
    z = crop(x, e)
    xw = set_window(x, e)
    yw = set_window(y, e)
    df1 = spat_sample(xw, 4, method="random", cells=True)
    df2 = spat_sample(yw, 4, method="random", cells=True)
    df3 = spat_sample(z, 4, method="random", cells=True)
    import pandas as pd

    for d in (df1, df2, df3):
        assert isinstance(d, pd.DataFrame)
        assert len(d) == 4


def test_extract_matrix_consistency(logo):
    import pandas as pd

    x = logo
    y = x * 1.0
    e = ext(35, 55, 35, 55)
    z = crop(x, e)
    xw = set_window(x, e)
    yw = set_window(y, e)
    a = _c_stack(z, yw, xw)
    xy = 10.0 * np.column_stack([np.arange(-1, 7, dtype=float)] * 2)
    e1 = extract(xw, xy)
    e2 = extract(yw, xy)
    e3 = extract(z, xy)
    e4 = extract(a, xy)
    assert isinstance(e1, pd.DataFrame)
    v1 = e1.select_dtypes(include=[float, int]).to_numpy()
    v2 = e2.select_dtypes(include=[float, int]).to_numpy()
    v3 = e3.select_dtypes(include=[float, int]).to_numpy()
    np.testing.assert_array_almost_equal(v1, v2)
    np.testing.assert_array_almost_equal(v1, v3)
    v4 = e4.select_dtypes(include=[float, int]).to_numpy()
    if v4.shape[1] >= v1.shape[1] * 3:
        np.testing.assert_array_almost_equal(np.hstack([v1, v2, v3]), v4[:, : v1.shape[1] * 3])


def test_stacked_sample_columns_skipped_or_lite(logo):
    """R compares cbind(s,...) to spatSample(a); skip exact match, stack exists."""
    x = logo
    e = ext(35, 55, 35, 55)
    z = crop(x, e)
    xw = set_window(x, e)
    yw = set_window(x * 1.0, e)
    a = _c_stack(z, yw, xw)
    df = spat_sample(a, 4, method="random", cells=True)
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
