"""
Tests ported from inst/tinytest/test_vrt.R

Covers VRT creation via sprc.make_vrt().
"""
import pytest

pytest.importorskip("tappa._terra")

import tappa as pt
from tappa.rast import rast
from tappa.values import set_values
from tappa.write import write_raster
from tappa.sprc import sprc


# https://github.com/rspatial/terra/issues/1410
class TestVrt:

    def test_vrt_separate(self, tmp_path):
        """VRT with -separate combines two single-layer rasters into 2 layers."""
        r = rast(ncols=100, nrows=100)
        r = set_values(r, [float(i + 1) for i in range(int(r.ncell()))])
        r2 = r * 2

        f1 = str(tmp_path / "r1.tif")
        f2 = str(tmp_path / "r2.tif")
        write_raster(r, f1)
        write_raster(r2, f2)

        vrt_file = str(tmp_path / "test.vrt")
        rc = sprc([f1, f2])
        vrt_path = rc.make_vrt(options=["-separate"], filename=vrt_file, overwrite=True)
        x = rast(vrt_path)

        assert x.nlyr() == 2
