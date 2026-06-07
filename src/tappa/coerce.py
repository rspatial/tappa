"""
coerce.py — type conversion between SpatRaster, SpatVector, SpatExtent,
            numpy arrays, and pandas DataFrames.
"""
from __future__ import annotations
from typing import Optional, Union
import numpy as np

from ._terra import SpatRaster, SpatVector, SpatExtent, SpatOptions
from ._helpers import messages, characterCRS


def _opt() -> SpatOptions:
    return SpatOptions()


# ---------------------------------------------------------------------------
# asPolygons
# ---------------------------------------------------------------------------

def asPolygons(
    x: Union[SpatRaster, SpatExtent, SpatVector],
    *,
    round: bool = True,
    aggregate: bool = True,
    values: bool = True,
    na_rm: bool = True,
    na_all: bool = False,
    extent: bool = False,
    digits: int = 0,
    crs: str = "",
) -> SpatVector:
    """
    Convert *x* to a SpatVector of polygons.

    Parameters
    ----------
    x : SpatRaster, SpatExtent, or SpatVector
    round : bool
        Round coordinates (SpatRaster only).
    aggregate : bool
        Dissolve adjacent cells with the same value (SpatRaster only).
    values : bool
        Transfer cell values as attributes (SpatRaster only).
    na_rm : bool
        Skip NA-valued cells (SpatRaster only).
    na_all : bool
        Treat cells as NA only when ALL layers are NA (SpatRaster only).
    extent : bool
        If True, return just the bounding box as a single polygon.
    digits : int
        Number of decimal places (SpatRaster only).
    crs : str
        CRS string (SpatExtent only).

    Returns
    -------
    SpatVector
    """
    if isinstance(x, SpatExtent):
        crs_str = characterCRS(crs, "asPolygons")
        v = SpatVector(x, crs_str)
        return messages(v, "asPolygons")
    if isinstance(x, SpatVector):
        if extent:
            from .extent import ext as make_ext
            from .crs import crs as get_crs
            e = make_ext(x)
            return asPolygons(e, crs=get_crs(x))
        if x.geomtype() == "points":
            x = asLines(x)
        xc = x.polygonize()
        return messages(xc, "asPolygons")
    # SpatRaster
    if extent:
        v = x.dense_extent(False, False)
        messages(x, "asPolygons")
        return messages(v, "asPolygons")
    opt = _opt()
    v = x.asPolygons(round, aggregate, values, na_rm, na_all, digits, opt)
    messages(x, "asPolygons")
    return messages(v, "asPolygons")


# ---------------------------------------------------------------------------
# asLines
# ---------------------------------------------------------------------------

def asLines(
    x: Union[SpatRaster, SpatExtent, SpatVector, np.ndarray],
    *,
    na_rm: bool = False,
    crs: str = "",
    segments: bool = False,
) -> SpatVector:
    """
    Convert *x* to a SpatVector of lines.

    Parameters
    ----------
    x : SpatRaster, SpatExtent, SpatVector, or ndarray
    na_rm : bool
        Skip NA cells (SpatRaster only).
    crs : str
        CRS string (ndarray / SpatExtent only).
    segments : bool
        Disaggregate into individual segments (ndarray only).

    Returns
    -------
    SpatVector
    """
    if isinstance(x, SpatExtent):
        return asLines(asPolygons(x, crs=crs))
    if isinstance(x, SpatRaster):
        if na_rm:
            p = asPolygons(x[0:1], round=False, aggregate=False, values=False, na_rm=True)
            return asLines(p)
        opt = _opt()
        v = x.asLines(opt)
        return messages(v, "asLines")
    if isinstance(x, SpatVector):
        xc = x.asLines()
        return messages(xc, "asLines")
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        v = SpatVector()
        crs_str = characterCRS(crs, "asLines")
        if arr.shape[1] == 2:
            nr = len(arr)
            v.setGeometry("lines",
                          list(range(0, nr)),
                          list(range(0, nr)),
                          arr[:, 0].tolist(),
                          arr[:, 1].tolist(),
                          [False] * nr)
            from .crs import set_crs
            set_crs(v, crs_str)
        elif arr.shape[1] == 4:
            v.setLinesStartEnd(arr, crs_str)
        else:
            raise ValueError("ndarray must have 2 or 4 columns")
        result = messages(v, "asLines")
        if segments:
            from .geom import disaggVect
            return disaggVect(result, segments=True)
        return result
    raise TypeError(f"asLines: unsupported type {type(x)}")


# ---------------------------------------------------------------------------
# asPoints
# ---------------------------------------------------------------------------

def asPoints(
    x: Union[SpatRaster, SpatExtent, SpatVector],
    *,
    values: bool = True,
    na_rm: bool = True,
    crs: str = "",
) -> SpatVector:
    """
    Convert *x* to a SpatVector of points.

    Parameters
    ----------
    x : SpatRaster, SpatExtent, or SpatVector
    values : bool
        Transfer cell values as attributes (SpatRaster only).
    na_rm : bool
        Skip NA cells (SpatRaster only).
    crs : str
        CRS string (SpatExtent only).

    Returns
    -------
    SpatVector
    """
    if isinstance(x, SpatExtent):
        return asPoints(asPolygons(x, crs=crs))
    if isinstance(x, SpatVector):
        xc = x.asPoints()
        return messages(xc, "asPoints")
    # SpatRaster
    opt = _opt()
    v = x.asPoints(values, na_rm, opt)
    messages(x, "asPoints")
    return messages(v, "asPoints")


# ---------------------------------------------------------------------------
# asArray / asMatrix  (raster → numpy)
# ---------------------------------------------------------------------------

def asArray(x: SpatRaster, na_value: float = float("nan")) -> np.ndarray:
    """
    Return all cell values as a 3-D numpy array.

    Parameters
    ----------
    x : SpatRaster
    na_value : float
        Value to use for NA cells.

    Returns
    -------
    numpy.ndarray, shape (nlyr, nrow, ncol).
    """
    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    v = np.array(x.readValues(0, nr, 0, nc), dtype=float)
    arr = v.reshape(nr * nc, nl, order='F').T.reshape(nl, nr, nc)
    if not np.isnan(na_value):
        arr = np.where(np.isnan(arr), na_value, arr)
    return arr


def asMatrix(
    x: Union[SpatRaster, SpatExtent],
) -> np.ndarray:
    """
    Return *x* as a 2-D numpy array.

    For SpatRaster, returns a (nrow, ncol) matrix of the first layer.
    For SpatExtent, returns a (2, 2) matrix [[xmin, xmax], [ymin, ymax]].

    Parameters
    ----------
    x : SpatRaster or SpatExtent

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(x, SpatExtent):
        return np.array([[x.xmin, x.xmax], [x.ymin, x.ymax]])
    nr, nc = x.nrow(), x.ncol()
    v = np.array(x.readValues(0, nr, 0, nc), dtype=float)
    return v[:nr * nc].reshape(nr, nc)


# ---------------------------------------------------------------------------
# asDataFrame  (raster → pandas)
# ---------------------------------------------------------------------------

def asDataFrame(
    x: SpatRaster,
    xy: bool = False,
    cells: bool = False,
    time: bool = False,
    na_rm: bool = False,
    wide: bool = True,
) -> "pd.DataFrame":
    """
    Convert a SpatRaster to a pandas DataFrame.

    Parameters
    ----------
    x : SpatRaster
    xy : bool
        Include x/y coordinate columns.
    cells : bool
        Include cell number column (0-based cell index).
    time : bool
        Include time column (if available).
    na_rm : bool
        Drop rows where all value columns are NA.
    wide : bool
        Wide format (one column per layer); only False is not yet supported.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for asDataFrame()")

    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    nms = list(x.names)

    v = np.array(x.readValues(0, nr, 0, nc), dtype=float).reshape(nr * nc, nl, order='F')

    col_data: dict = {}
    if cells:
        col_data["cell"] = np.arange(0, nr * nc)
    if xy:
        xs = np.tile(np.linspace(x.xmin() + x.xres() / 2,
                                  x.xmax() - x.xres() / 2, nc), nr)
        ys = np.repeat(np.linspace(x.ymax() - x.yres() / 2,
                                    x.ymin() + x.yres() / 2, nr), nc)
        col_data["x"] = xs
        col_data["y"] = ys
    for i, name in enumerate(nms):
        col_data[name] = v[:, i]

    df = pd.DataFrame(col_data)
    if na_rm:
        value_cols = nms
        df = df.dropna(subset=value_cols, how="all").reset_index(drop=True)
    return df
