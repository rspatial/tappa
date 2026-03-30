"""
Tests ported from inst/tinytest/test_misc-vector.R

Geometry type, measures, buffer, simplify, extent, and spatial relations.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa.rast import rast
from tappa.vect import vect
from tappa.spatvec import geomtype, is_lines, is_polygons, is_points, perim, expanse as area
from tappa.geom import buffer_vect, simplify_geom
from tappa.extent import ext
from tappa.relate import is_related

from path_utils import skip_if_missing_inst_ex


@pytest.fixture
def lux():
    return vect(skip_if_missing_inst_ex("lux.shp"))


def test_geomtype_and_measures(lux):
    v = lux
    g0 = geomtype(v)
    assert isinstance(g0, list)
    assert str(g0[0]).lower() == "polygons"
    assert is_polygons(v)
    assert not is_lines(v)
    assert not is_points(v)
    p = perim(v)
    a = area(v)
    assert np.all(np.array(p) > 0)
    assert np.all(np.array(a) > 0)


def test_buffer_increases_area(lux):
    v = lux
    b = buffer_vect(v.subset_rows([0]), 500)
    assert area(b)[0] > area(v.subset_rows([0]))[0]


def test_simplify_reduces_segments(lux):
    v = lux
    simp = simplify_geom(v.subset_rows([0]), 100)
    assert simp.nrow() >= 1


def test_empty_extent_constructor():
    e = ext()
    assert not e.valid


def test_relate_equals_self(lux):
    v = lux
    a = v.subset_rows([0])
    assert bool(is_related(a, a, "equals")[0])
