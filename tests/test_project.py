"""
Tests for project() (SpatVector, SpatRaster) and projPipelines().
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa.rast import rast
from tappa.vect import vect
from tappa.dispatch import project
from tappa.crs import crs, projPipelines
from tappa.values import values

from path_utils import skip_if_missing_inst_ex


# ── projPipelines ────────────────────────────────────────────────────────

def test_proj_pipelines_basic():
    df = projPipelines("EPSG:4326", "EPSG:3857")
    assert df is not None
    assert len(df) > 0
    assert "definition" in df.columns
    assert "accuracy" in df.columns
    for d in df["definition"]:
        assert d.startswith("+")


def test_proj_pipelines_with_aoi():
    df = projPipelines("EPSG:4326", "EPSG:32632", AOI=[5, 45, 15, 55])
    assert df is not None
    assert len(df) > 0


# ── SpatVector project ────────────────────────────────────────────────────

@pytest.fixture
def lux():
    return vect(skip_if_missing_inst_ex("lux.shp"))


def test_project_vector_basic(lux):
    v2 = project(lux, "EPSG:3857")
    assert crs(v2, proj4=True) != crs(lux, proj4=True)


def test_project_vector_allow_ballpark_false(lux):
    v2 = project(lux, "EPSG:3857", allow_ballpark=False)
    assert crs(v2, proj4=True) != crs(lux, proj4=True)


def test_project_vector_pipeline(lux):
    pipes = projPipelines("EPSG:4326", "EPSG:3857")
    if pipes is None or len(pipes) == 0:
        pytest.skip("no pipeline available")
    pipe = pipes["definition"].iloc[0]
    v2 = project(lux, "EPSG:3857", pipeline=pipe)
    e = v2.extent()
    assert not np.isnan(e.vector[0])


# ── SpatRaster project ───────────────────────────────────────────────────

@pytest.fixture
def elev():
    return rast(skip_if_missing_inst_ex("elev.tif"))


def test_project_raster_basic(elev):
    r2 = project(elev, "EPSG:3857")
    assert crs(r2, proj4=True) != crs(elev, proj4=True)


def test_project_raster_allow_ballpark_false(elev):
    r2 = project(elev, "EPSG:3857", allow_ballpark=False)
    v = values(r2)
    assert not np.all(np.isnan(v))


def test_project_raster_pipeline(elev):
    from_crs = crs(elev)
    pipes = projPipelines(from_crs, "EPSG:3857")
    if pipes is None or len(pipes) == 0:
        pytest.skip("no pipeline available")
    pipe = pipes["definition"].iloc[0]
    r2 = project(elev, "EPSG:3857", pipeline=pipe)
    v = values(r2)
    assert not np.all(np.isnan(v))
