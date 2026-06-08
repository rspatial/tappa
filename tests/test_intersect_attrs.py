"""Tests for attribute handling in vector intersect (terra parity)."""
import numpy as np
import pytest
import tappa as pt
from tappa.dispatch import intersect
from tappa.rasterize import rasterize
from tappa.rast import rast


def test_intersect_keeps_duplicate_column_names():
    pts = pt.vect(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        crs="+proj=longlat +datum=WGS84",
    )
    df = pt.vect_as_df(pts).copy()
    df["COUNTRY"] = ["LEFT", "LEFT"]
    pts = pt.set_values(pts, df)

    pol = pt.vect(
        "POLYGON((-1 -1, 2 -1, 2 2, -1 2, -1 -1))",
        crs="+proj=longlat +datum=WGS84",
    )
    pdf = pt.vect_as_df(pol).copy()
    pdf["COUNTRY"] = ["RIGHT"]
    pol = pt.set_values(pol, pdf)

    out = intersect(pts, pol)
    # terra::intersect also calls make_unique_names() in the C++ core.
    cols = list(pt.vect_as_df(out).columns)
    assert cols == ["COUNTRY_1", "COUNTRY_2"]

    # R tutorial: names(vv)[1] <- "ptCountry"
    renamed = cols.copy()
    renamed[0] = "ptCountry"
    frame = pt.vect_as_df(out)
    frame.columns = renamed
    assert list(frame.columns) == ["ptCountry", "COUNTRY_2"]


def test_rasterize_count_with_string_field():
    pts = pt.vect(
        np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
        crs="+proj=longlat +datum=WGS84",
    )
    df = pt.vect_as_df(pts).copy()
    df["SPECIES"] = ["sp_a", "sp_b", None]
    pts = pt.set_values(pts, df)

    template = rast(xmin=0, xmax=1, ymin=0, ymax=1, ncols=1, nrows=1,
                    crs="+proj=longlat +datum=WGS84")
    out = rasterize(pts, template, field="SPECIES", fun="count")
    val = float(np.array(out.readValues(0, 1, 0, 1), dtype=float)[0])
    assert val == 3.0

    out_na = rasterize(pts, template, field="SPECIES", fun="count", na_rm=True)
    val_na = float(np.array(out_na.readValues(0, 1, 0, 1), dtype=float)[0])
    assert val_na == 2.0
