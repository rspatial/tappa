"""
Tests for unified generics: buffer(), project(), crop(), mask().
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa.rast import rast
from tappa.vect import vect
from tappa.dispatch import buffer, project, intersect
from tappa.distance import distance
from tappa.write import write
from tappa.merge import merge
from tappa.subset import subset
from tappa.names import names, set_names
from tappa.values import values, set_values
from tappa.aggregate import aggregate
from tappa.generics import flip, rotate, shift, rescale, disagg
from tappa.generics import crop, mask
from tappa.crs import crs
from tappa.spatvec import expanse as area
from tappa.extent import ext

from path_utils import skip_if_missing_inst_ex


def _ext_vec(obj):
    return [obj.xmin(), obj.xmax(), obj.ymin(), obj.ymax()]


@pytest.fixture
def lux():
    return vect(skip_if_missing_inst_ex("lux.shp"))


@pytest.fixture
def elev():
    return rast(skip_if_missing_inst_ex("elev.tif"))


def test_buffer_vector(lux):
    v = lux.subset_rows([0])
    b = buffer(v, 500)
    assert area(b)[0] > area(v)[0]
    assert v.buffer(500).nrow() == b.nrow()


def test_buffer_raster_method(elev):
    r2 = elev.buffer(100)
    assert r2 is not None
    assert r2.nlyr() == elev.nlyr()


def test_project_vector(lux):
    v = project(lux, "EPSG:3857")
    assert crs(v, proj4=True) != crs(lux, proj4=True)
    assert lux.project("EPSG:3857") is not None


def test_project_raster(elev):
    r = project(elev, "EPSG:3857")
    assert crs(r, proj4=True) != crs(elev, proj4=True)
    assert elev.project("EPSG:3857") is not None


def test_crop_vector(lux):
    e = ext(6.0, 6.5, 49.5, 50.0)
    c = crop(lux, e)
    assert c.nrow() >= 0
    assert lux.crop(e).nrow() == c.nrow()


def test_mask_vector(lux, elev):
    m = mask(lux, elev)
    assert m.nrow() <= lux.nrow()
    assert lux.mask(elev).nrow() == m.nrow()


def test_crop_raster_unchanged():
    r = rast(nrows=10, ncols=10, xmin=0, xmax=10, ymin=0, ymax=10)
    e = ext(0, 5, 0, 5)
    np.testing.assert_array_almost_equal(_ext_vec(crop(r, e)), [0, 5, 0, 5])


def test_dispatch_type_error():
    with pytest.raises(TypeError, match="buffer"):
        buffer("not spatial", 1)
    with pytest.raises(TypeError, match="project"):
        project(42, "EPSG:4326")
    with pytest.raises(TypeError, match="crop"):
        crop([], ext(0, 1, 0, 1))
    with pytest.raises(TypeError, match="mask"):
        mask(None, ext(0, 1, 0, 1))


def test_top_level_exports():
    assert pt.buffer is buffer
    assert pt.project is project
    assert pt.intersect is intersect
    assert pt.distance is distance
    assert pt.write is write
    assert pt.merge is merge
    assert pt.subset is subset
    assert pt.names is names
    assert pt.values is values
    assert pt.aggregate is aggregate


def test_names_unified(lux, elev):
    assert isinstance(names(lux), list)
    assert isinstance(names(elev), list)
    assert lux.names == names(lux)


def test_values_vector(lux):
    df = values(lux)
    assert hasattr(df, "columns")


def test_intersect_vector(lux):
    e = ext(6.0, 6.5, 49.5, 50.0)
    i = intersect(lux, e)
    assert i.nrow() <= lux.nrow()


def test_flip_shift_rescale_methods(lux):
    assert flip(lux).nrow() == lux.nrow()
    assert shift(lux, 0.01, 0.01).nrow() == lux.nrow()
    assert rescale(lux, 0.9).nrow() == lux.nrow()


def test_disagg_vector(lux):
    d = disagg(lux)
    assert d.nrow() >= lux.nrow()


def test_distance_vector_self(lux):
    d = distance(lux)
    assert d.shape[0] == lux.nrow()
