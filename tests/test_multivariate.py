"""
Tests ported from inst/tinytest/test_multivariate.R

Covers k_means (skipped if not available) and layerCor.
"""
import numpy as np
import pytest
import tappa as pt
from tappa.rast import rast

from path_utils import skip_if_missing_inst_ex

from tappa.stats import layerCor


def find_logo():
    return skip_if_missing_inst_ex("logo.tif")


def test_layercor_string_vs_callable():
    """layerCor(x, 'cor') and layerCor(x, cor) should give equivalent results."""
    f = find_logo()
    x = rast(f)

    import numpy as np
    a = layerCor(x, "cor")
    b = layerCor(x, np.corrcoef)

    # a["correlation"] from string should equal b from callable
    if isinstance(a, dict):
        a_mat = a["correlation"]
    else:
        a_mat = a
    if isinstance(b, dict):
        b_mat = b["correlation"]
    else:
        b_mat = b

    np.testing.assert_array_almost_equal(
        np.array(a_mat), np.array(b_mat), decimal=5
    )


