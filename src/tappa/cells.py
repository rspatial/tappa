"""
cells.py — cell index operations for SpatRaster.

All cell, row, and column indices exposed here are **0-based**, matching the
C++ core and NumPy-style Python conventions.
"""
from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

from ._terra import SpatRaster, SpatVector, SpatExtent, SpatOptions
from ._helpers import messages


def _as_cell_list(cell: Union[int, float, np.integer, List, np.ndarray]) -> List[int]:
    """Coerce *cell* to a list of 0-based cell indices."""
    if isinstance(cell, (list, tuple)):
        return [int(c) for c in cell]
    if isinstance(cell, np.ndarray):
        return [int(c) for c in np.asarray(cell, dtype=np.int64).ravel()]
    return [int(cell)]


def _opt() -> SpatOptions:
    return SpatOptions()


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def cells(
    x: SpatRaster,
    y: Optional[Union[float, List[float], SpatVector, SpatExtent]] = None,
    *,
    method: str = "simple",
    weights: bool = False,
    exact: bool = False,
    touches: Optional[bool] = None,
    small: bool = True,
    pairs: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Return cell numbers for non-NA cells or cells matching given criteria.

    Parameters
    ----------
    x : SpatRaster
    y : float, list, SpatVector, or SpatExtent, optional
        - If absent: all cell indices ``0 .. ncell(x)-1`` when there are no
          values; otherwise non-NA cell indices (0-based).
        - If a scalar or list: find cells whose values match *y*.
        - If a SpatVector: find cells overlapping *y*.
        - If a SpatExtent: find cells within *y*.
    method : str
        ``"simple"`` or ``"bilinear"`` (for SpatVector point lookups).
    weights : bool
        Include fractional coverage weights (SpatVector polygons only).
    exact : bool
        Compute exact fractional weights.
    touches : bool, optional
        Include cells touching the polygon boundary.
    small : bool
        Include small polygons.
    pairs : bool
        If *y* is numeric, return ``(cell, value)`` pairs.

    Returns
    -------
    numpy.ndarray (1-D cell indices, 0-based), or
    list of arrays (one per layer when *y* is numeric), or
    numpy.ndarray with columns [ID, cell, …] when *y* is a SpatVector.
    """
    opt = _opt()

    if y is None:
        if x.hasValues:
            raw = x.cells_notna_novalues(opt)
            return np.array(raw, dtype=int)
        n = x.ncell()
        return np.arange(0, n, dtype=int)

    if isinstance(y, SpatExtent):
        raw = x.extCells(y.pntr) if hasattr(y, "pntr") else x.extCells(y)
        return np.array(raw, dtype=int)

    if isinstance(y, SpatVector):
        if touches is None:
            from .vect import is_lines

            touches = is_lines(y)
        raw = x.vectCells(y, touches, small, method, weights, exact, opt)
        if y.geomtype() == "points":
            if method == "bilinear":
                m = np.array(raw, dtype=float).reshape(y.nrow(), -1)
                ids = np.arange(0, y.nrow(), dtype=int).reshape(-1, 1)
                m = np.hstack([ids, m])
                return m
            else:
                m = np.array(raw, dtype=float).reshape(y.nrow(), -1)
                ids = np.arange(0, y.nrow(), dtype=int).reshape(-1, 1)
                m = np.hstack([ids, m])
                return m
        else:
            ncols = 3 if (weights or exact) else 2
            m = np.array(raw, dtype=float).reshape(-1, ncols)
            return m

    # Numeric value matching
    if isinstance(y, (int, float)):
        y = [float(y)]
    else:
        y = [float(v) for v in y]
    raw_list = x.is_in_cells(y, pairs, opt)
    result = messages(x, "cells")
    if pairs:
        out = []
        for arr in raw_list:
            m = np.array(arr, dtype=float).reshape(-1, 2)
            out.append(m)
        return out
    else:
        return [np.array(arr, dtype=int) for arr in raw_list]


# ---------------------------------------------------------------------------
# Row / col / cell conversions
# ---------------------------------------------------------------------------

def row_from_y(x: SpatRaster, y: Union[float, List[float]]) -> np.ndarray:
    """Return row indices (0-based) for y-coordinates."""
    if isinstance(y, (int, float)):
        y = [float(y)]
    return np.array(x.rowFromY(y), dtype=int)


def col_from_x(x: SpatRaster, xcoord: Union[float, List[float]]) -> np.ndarray:
    """Return column indices (0-based) for x-coordinates."""
    if isinstance(xcoord, (int, float)):
        xcoord = [float(xcoord)]
    return np.array(x.colFromX(xcoord), dtype=int)


def cell_from_xy(
    x: SpatRaster,
    xy: Union[float, List, np.ndarray],
    y: Union[float, List, np.ndarray, None] = None,
) -> np.ndarray:
    """
    Return cell numbers for x/y coordinate pairs.

    Parameters
    ----------
    x : SpatRaster
    xy : array-like of shape (n, 2), or a single x-coordinate (or list of x's)
        when used with the *y* argument. Columns of the matrix form are
        ``[x_coord, y_coord]``.
    y : float or array-like, optional
        Y-coordinate(s) when *xy* is the x-coordinate(s).

    Returns
    -------
    numpy.ndarray, dtype float64
        0-based cell numbers.  Invalid coordinates (outside extent or NaN
        inputs) are ``nan``.
    """
    if y is not None:
        xs = np.atleast_1d(np.asarray(xy, dtype=float)).ravel()
        ys = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        if xs.shape != ys.shape:
            raise ValueError("cell_from_xy: x and y must have the same length")
    else:
        arr = np.asarray(xy, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xs = arr[:, 0]
        ys = arr[:, 1]
    raw = x.cellFromXY(xs.tolist(), ys.tolist(), float("nan"))
    return np.array(raw, dtype=float)


def cell_from_row_col(
    x: SpatRaster,
    row: Union[int, List[int]],
    col: Union[int, List[int]],
) -> np.ndarray:
    """Return cell numbers for row/column index pairs (0-based)."""
    if isinstance(row, int):
        row = [row]
    if isinstance(col, int):
        col = [col]
    raw = x.cellFromRowCol(list(row), list(col))
    return np.array(raw, dtype=int)


def xy_from_cell(x: SpatRaster, cell: Union[int, float, np.integer, List[int], np.ndarray]) -> np.ndarray:
    """
    Return x/y coordinates for cell numbers.

    Parameters
    ----------
    x : SpatRaster
    cell : int or list of int (0-based)

    Returns
    -------
    numpy.ndarray, shape (n, 2), columns [x, y].
    """
    cell = _as_cell_list(cell)
    coords = x.xyFromCell([float(c) for c in cell])
    return np.array(coords, dtype=float).reshape(-1, 2)


def row_col_from_cell(
    x: SpatRaster,
    cell: Union[int, float, np.integer, List[int], np.ndarray],
) -> np.ndarray:
    """
    Return row and column indices for cell numbers.

    Parameters
    ----------
    x : SpatRaster
    cell : int or list (0-based)

    Returns
    -------
    numpy.ndarray, shape (n, 2), columns [row, col] (0-based).
    """
    cell = _as_cell_list(cell)
    cell_f = [float(c) for c in cell]
    rc = x.rowColFromCell(cell_f)
    arr = np.array(rc, dtype=int).reshape(-1, 2)
    return arr
