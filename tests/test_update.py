"""
Tests for tappa.update — on-disk metadata and cell-value updates.

Covers:
  - update cell values (single layer, multi-layer, specific layer, broadcast)
  - update metadata (names, crs, extent)
  - in-memory raster warning
"""
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("tappa._terra")

from tappa.rast import rast
from tappa.values import set_values, values
from tappa.write import write_raster, update
from tappa.names import set_names_inplace


class TestUpdateValues:

    def test_single_layer(self, tmp_path):
        """Update specific cells in a single-layer raster."""
        r = rast(nrows=10, ncols=10)
        r = set_values(r, list(range(1, 101)))
        f = str(tmp_path / "r1.tif")
        x = write_raster(r, f)

        update(x, cells=[1, 50, 100], values=[999, 888, 777])
        y = rast(f)
        v = values(y)
        assert v[0, 0] == 999
        assert v[49, 0] == 888
        assert v[99, 0] == 777
        assert v[1, 0] == 2  # unchanged

    def test_all_layers_broadcast(self, tmp_path):
        """Same values broadcast to all layers."""
        r = rast(nrows=5, ncols=5, nlyrs=3)
        r = set_values(r, np.arange(1, 76, dtype=float).tolist())
        f = str(tmp_path / "r2.tif")
        x = write_raster(r, f, datatype="FLT4S")

        update(x, cells=[1, 13], values=[-5, -10])
        y = rast(f)
        v = values(y)
        assert np.all(v[0, :] == -5)
        assert np.all(v[12, :] == -10)
        assert v[1, 0] == 2  # unchanged

    def test_specific_layer(self, tmp_path):
        """Update only layer 2 of a 3-layer raster."""
        r = rast(nrows=5, ncols=5, nlyrs=3)
        r = set_values(r, np.arange(1, 76, dtype=float).tolist())
        f = str(tmp_path / "r3.tif")
        x = write_raster(r, f, datatype="FLT4S")

        update(x, cells=[1, 25], values=[0, 0], layer=2)
        y = rast(f)
        v = values(y)
        assert list(v[0, :])  == [1, 0, 51]
        assert list(v[24, :]) == [25, 0, 75]

    def test_per_layer_values(self, tmp_path):
        """Provide cs*nlyrs values in layer-major order."""
        r = rast(nrows=5, ncols=5, nlyrs=2)
        r = set_values(r, np.arange(1, 51, dtype=float).tolist())
        f = str(tmp_path / "r4.tif")
        x = write_raster(r, f, datatype="FLT4S")

        update(x, cells=[1, 2], values=[100, 200, 300, 400])
        y = rast(f)
        v = values(y)
        assert list(v[0, :]) == [100, 300]
        assert list(v[1, :]) == [200, 400]

    def test_single_value_broadcast(self, tmp_path):
        """A single value is recycled to all cells and layers."""
        r = rast(nrows=5, ncols=5)
        r = set_values(r, np.arange(1, 26, dtype=float).tolist())
        f = str(tmp_path / "r5.tif")
        x = write_raster(r, f, datatype="FLT4S")

        update(x, cells=[1, 2, 3, 4, 5], values=[0])
        y = rast(f)
        v = values(y)
        assert np.all(v[:5, 0] == 0)
        assert v[5, 0] == 6  # unchanged

    def test_nan_to_nodata(self, tmp_path):
        """NaN values are written as NoData."""
        r = rast(nrows=5, ncols=5)
        r = set_values(r, np.arange(1, 26, dtype=float).tolist())
        f = str(tmp_path / "r6.tif")
        x = write_raster(r, f, datatype="FLT4S")

        update(x, cells=[1, 13], values=[float("nan"), float("nan")])
        y = rast(f)
        v = values(y)
        assert math.isnan(v[0, 0])
        assert math.isnan(v[12, 0])


class TestUpdateMeta:

    def test_update_names(self, tmp_path):
        """Band names persist after update(names=True)."""
        r = rast(nrows=5, ncols=5, nlyrs=2)
        r = set_values(r, np.arange(1, 51, dtype=float).tolist())
        f = str(tmp_path / "meta1.tif")
        x = write_raster(r, f, datatype="FLT4S")

        set_names_inplace(x, ["A", "B"])
        update(x, names=True)
        y = rast(f)
        assert list(y.names) == ["A", "B"]

    def test_update_crs(self, tmp_path):
        """CRS persists after update(crs=True)."""
        r = rast(nrows=5, ncols=5)
        r = set_values(r, list(range(1, 26)))
        f = str(tmp_path / "meta2.tif")
        x = write_raster(r, f)

        x.set_crs("+proj=utm +zone=1")
        update(x, crs=True)
        y = rast(f)
        assert "UTM zone 1N" in y.get_crs("wkt")

    def test_update_extent(self, tmp_path):
        """Extent persists after update(extent=True)."""
        from tappa._terra import SpatExtent
        r = rast(nrows=5, ncols=5, xmin=0, xmax=5, ymin=0, ymax=5)
        r = set_values(r, list(range(1, 26)))
        f = str(tmp_path / "meta3.tif")
        x = write_raster(r, f)

        new_ext = SpatExtent()
        new_ext.vector = [-1, 6, -1, 6]
        x.extent = new_ext
        update(x, extent=True)
        y = rast(f)
        assert y.extent.vector[0] == pytest.approx(-1.0)


class TestUpdateInMemory:

    def test_in_memory_values_warns(self):
        """update() on in-memory raster issues a warning."""
        r = rast(nrows=5, ncols=5)
        r = set_values(r, list(range(1, 26)))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            update(r, cells=[1], values=[99])
            assert any("in-memory" in str(warning.message) for warning in w)

    def test_in_memory_meta_warns(self):
        """update(names=True) on in-memory raster issues a warning."""
        r = rast(nrows=5, ncols=5)
        r = set_values(r, list(range(1, 26)))
        set_names_inplace(r, ["test"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            update(r, names=True)
            assert any("in-memory" in str(warning.message) for warning in w)
