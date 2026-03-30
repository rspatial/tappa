"""
stats.py — row/column statistics and value matching for SpatRaster.

Covers functionality from rowSums.R, match.R, autocor.R, layerCor.R.
"""
from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

from ._terra import SpatRaster, SpatOptions
from ._helpers import messages, spatoptions


def _opt() -> SpatOptions:
    return SpatOptions()


def _read_values_layer_matrix(x: SpatRaster) -> np.ndarray:
    """
    Read all cell values as (nrow * ncol, nlyr), Fortran order (aligned with R).

    GDAL-backed rasters require readStart before readValues.
    """
    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    x.readStart()
    try:
        raw = x.readValues(0, nr, 0, nc)
    finally:
        x.readStop()
    return np.array(raw, dtype=float).reshape(nr * nc, nl, order="F")


# ---------------------------------------------------------------------------
# Row / column statistics
# ---------------------------------------------------------------------------

def row_sums(x: SpatRaster, na_rm: bool = False) -> np.ndarray:
    """
    Sum of each row across columns, returned separately per layer.

    Parameters
    ----------
    x : SpatRaster
    na_rm : bool

    Returns
    -------
    numpy.ndarray, shape (nrow, nlyr).
    """
    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    vals = _read_values_layer_matrix(x)
    vals_3d = vals.reshape(nr, nc, nl)
    if na_rm:
        return np.nansum(vals_3d, axis=1)
    return np.sum(vals_3d, axis=1)


def col_sums(x: SpatRaster, na_rm: bool = False) -> np.ndarray:
    """
    Sum of each column across rows, returned separately per layer.

    Parameters
    ----------
    x : SpatRaster
    na_rm : bool

    Returns
    -------
    numpy.ndarray, shape (ncol, nlyr).
    """
    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    vals = np.array(x.readValues(0, nr, 0, nc), dtype=float).reshape(nr * nc, nl, order='F')
    vals_3d = vals.reshape(nr, nc, nl)
    if na_rm:
        return np.nansum(vals_3d, axis=0)
    return np.sum(vals_3d, axis=0)


def row_means(x: SpatRaster, na_rm: bool = False) -> np.ndarray:
    """
    Mean of each row across columns, returned separately per layer.

    Parameters
    ----------
    x : SpatRaster
    na_rm : bool

    Returns
    -------
    numpy.ndarray, shape (nrow, nlyr).
    """
    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    vals = _read_values_layer_matrix(x)
    vals_3d = vals.reshape(nr, nc, nl)
    if na_rm:
        return np.nanmean(vals_3d, axis=1)
    return np.mean(vals_3d, axis=1)


def col_means(x: SpatRaster, na_rm: bool = False) -> np.ndarray:
    """
    Mean of each column across rows, returned separately per layer.

    Parameters
    ----------
    x : SpatRaster
    na_rm : bool

    Returns
    -------
    numpy.ndarray, shape (ncol, nlyr).
    """
    nr, nc = x.nrow(), x.ncol()
    nl = x.nlyr()
    vals = np.array(x.readValues(0, nr, 0, nc), dtype=float).reshape(nr * nc, nl, order='F')
    vals_3d = vals.reshape(nr, nc, nl)
    if na_rm:
        return np.nanmean(vals_3d, axis=0)
    return np.mean(vals_3d, axis=0)


# ---------------------------------------------------------------------------
# match / is_in
# ---------------------------------------------------------------------------

def match_rast(
    x: SpatRaster,
    table: List,
    nomatch: float = float("nan"),
) -> SpatRaster:
    """
    Return the position of each cell's value in *table*.

    Parameters
    ----------
    x : SpatRaster
    table : list
        Values to match against.
    nomatch : float
        Value to use when there is no match (default: NaN).

    Returns
    -------
    SpatRaster
    """
    import numpy as np
    from .app import app

    table_u = list(dict.fromkeys(table))
    table_np = np.array(table_u, dtype=float)

    def _match(v: np.ndarray) -> np.ndarray:
        out = np.full_like(v, nomatch, dtype=float)
        for idx, t in enumerate(table_np):
            mask = v == t
            out[mask] = idx + 1
        return out

    return app(x, _match)


def is_in(
    x: SpatRaster,
    table: List,
) -> SpatRaster:
    """
    Return a binary raster: 1 where cell values are in *table*, 0 otherwise.

    Parameters
    ----------
    x : SpatRaster
    table : list of values

    Returns
    -------
    SpatRaster
    """
    table_u = list(dict.fromkeys([t for t in table if t is not None]))
    opt = _opt()
    xc = x.is_in([float(t) for t in table_u], opt)
    return messages(xc, "is_in")


# ---------------------------------------------------------------------------
# autocor — spatial autocorrelation (Moran's I / Geary's C)
# Mirrors R autocor.R: builds entirely from focal + global arithmetic.
# ---------------------------------------------------------------------------

def _raster_global_nansum(x: SpatRaster) -> float:
    mat = _read_values_layer_matrix(x)
    return float(np.nansum(mat[:, 0]))


def _raster_global_mean(x: SpatRaster) -> float:
    mat = _read_values_layer_matrix(x)
    return float(np.nanmean(mat[:, 0]))


def _raster_global_sd(x: SpatRaster) -> float:
    mat = _read_values_layer_matrix(x)
    vals = mat[:, 0]
    return float(np.nanstd(vals, ddof=1))


def _raster_count_valid(x: SpatRaster) -> int:
    mat = _read_values_layer_matrix(x)
    return int(np.sum(np.isfinite(mat[:, 0])))


def _autocor_weight_matrix(w: str) -> np.ndarray:
    t = w.lower()
    if t == "queen":
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    if t == "rook":
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    raise ValueError(f"unknown autocor weight name: {w!r}")


def _get_filter_matrix(w: np.ndarray, *, warn: bool = True) -> np.ndarray:
    """Return a valid focal weight matrix with centre cell zeroed (R .getFilter)."""
    import warnings as _warnings
    arr = np.asarray(w, dtype=float)
    if arr.ndim != 2:
        raise ValueError("weight matrix must be 2-D")
    if arr.shape[0] % 2 == 0 or arr.shape[1] % 2 == 0:
        raise ValueError("dimensions of weights matrix (filter) must be odd")
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    if arr[cy, cx] != 0:
        if warn:
            _warnings.warn("central cell of weights matrix (filter) was set to zero")
        arr = arr.copy()
        arr[cy, cx] = 0.0
    return arr


def _resolve_weights_moran(w: Optional[Union[str, np.ndarray]]) -> np.ndarray:
    if w is None or (isinstance(w, str) and w.lower() == "queen"):
        return _autocor_weight_matrix("queen")
    if isinstance(w, str):
        return _autocor_weight_matrix(w)
    return _get_filter_matrix(np.asarray(w, dtype=float))


def _global_moran(
    x: SpatRaster, w: Union[str, np.ndarray, None], filename: str, overwrite: bool
) -> float:
    from .arith import _rast_arith_numb, _rast_arith_rast, is_na
    from .math import ifel
    from .focal import focal

    wmat = _resolve_weights_moran(w)
    mu = _raster_global_mean(x)
    z = messages(_rast_arith_numb(x, mu, "-"), "autocor")

    wz = messages(focal(z, wmat, fun="sum", na_rm=True, na_policy="omit"), "autocor")
    wZiZj = messages(_rast_arith_rast(wz, z, "*"), "autocor")
    wZiZj_sum = _raster_global_nansum(wZiZj)

    z2 = messages(_rast_arith_rast(z, z, "*"), "autocor")
    z2_sum = _raster_global_nansum(z2)
    if z2_sum == 0.0:
        return float("nan")

    n_valid = _raster_count_valid(x)
    zz = messages(ifel(is_na(x), float("nan"), 1.0), "autocor")
    W_rast = messages(focal(zz, wmat, fun="sum", na_rm=True, na_policy="omit"), "autocor")
    W_sum = _raster_global_nansum(W_rast)
    if W_sum == 0.0:
        return float("nan")

    ns0 = n_valid / W_sum
    return float(ns0 * wZiZj_sum / z2_sum)


def _global_geary(
    x: SpatRaster, w: Union[str, np.ndarray, None], filename: str, overwrite: bool
) -> float:
    from .arith import _rast_arith_numb, _rast_arith_rast, is_na
    from .math import ifel
    from .focal import focal

    wmat = _resolve_weights_moran(w)
    wmat_f = _get_filter_matrix(wmat, warn=False)
    n_w = wmat_f.shape[0] * wmat_f.shape[1]
    center_idx = n_w // 2

    def _geary_fun(vals: np.ndarray) -> float:
        return float(np.nansum((vals - vals[center_idx]) ** 2))

    f = messages(focal(x, wmat_f, fun=_geary_fun, na_rm=True), "autocor")
    e_ij = _raster_global_nansum(f)

    n_valid = _raster_count_valid(x)
    xx = messages(ifel(is_na(x), float("nan"), 1.0), "autocor")
    W_rast = messages(focal(xx, wmat_f, fun="sum", na_rm=True), "autocor")
    W_sum = _raster_global_nansum(W_rast)

    mu = _raster_global_mean(x)
    xdiff = messages(_rast_arith_numb(x, mu, "-"), "autocor")
    xdiff2 = messages(_rast_arith_rast(xdiff, xdiff, "*"), "autocor")
    z = 2.0 * W_sum * _raster_global_nansum(xdiff2)
    if z == 0.0:
        return float("nan")

    return float((n_valid - 1) * e_ij / z)


def _local_moran(
    x: SpatRaster, w: Union[str, np.ndarray, None], filename: str, overwrite: bool
) -> SpatRaster:
    from .arith import _rast_arith_numb, _rast_arith_rast, is_na
    from .math import ifel
    from .focal import focal

    wmat = _resolve_weights_moran(w)
    mu = _raster_global_mean(x)
    z = messages(_rast_arith_numb(x, mu, "-"), "autocor")

    zz = messages(ifel(is_na(x), float("nan"), 1.0), "autocor")
    W = messages(focal(zz, wmat, fun="sum", na_rm=True), "autocor")
    fz = messages(focal(z, wmat, fun="sum", na_rm=True), "autocor")
    lz = messages(_rast_arith_rast(fz, W, "/"), "autocor")

    s2 = _raster_global_sd(x) ** 2
    if s2 == 0.0:
        return messages(_rast_arith_numb(x, float("nan"), "*"), "autocor")

    z_scaled = messages(_rast_arith_numb(z, s2, "/"), "autocor")
    return messages(_rast_arith_rast(z_scaled, lz, "*"), "autocor")


def _local_geary(
    x: SpatRaster, w: Union[str, np.ndarray, None], filename: str, overwrite: bool
) -> SpatRaster:
    from .focal import focal

    wmat = _resolve_weights_moran(w)
    wmat_f = _get_filter_matrix(wmat, warn=False)
    n_w = wmat_f.shape[0] * wmat_f.shape[1]
    center_idx = n_w // 2

    def _geary_fun(vals: np.ndarray) -> float:
        return float(np.nansum((vals - vals[center_idx]) ** 2))

    e_ij = messages(focal(x, wmat_f, fun=_geary_fun, na_rm=True), "autocor")
    s2 = _raster_global_sd(x) ** 2
    if s2 == 0.0:
        from .arith import _rast_arith_numb
        return messages(_rast_arith_numb(x, float("nan"), "*"), "autocor")

    from .arith import _rast_arith_numb
    return messages(_rast_arith_numb(e_ij, s2, "/"), "autocor")


def autocor(
    x: SpatRaster,
    w: Optional[Union[str, "np.ndarray"]] = None,
    global_: bool = True,
    method: str = "moran",
    filename: str = "",
    overwrite: bool = False,
) -> Union[float, SpatRaster]:
    """
    Compute spatial autocorrelation for *x*.

    Parameters
    ----------
    x : SpatRaster
        Single-layer raster.
    w : str or array, optional
        Weights matrix.  ``"queen"`` (8-neighbour, default), ``"rook"``
        (4-neighbour), or a custom 2-D weight matrix.
    global_ : bool
        If True, return a scalar (global Moran's I or Geary's C).
        If False, return a local SpatRaster.
    method : str
        ``"moran"`` (default) or ``"geary"``.
    filename : str
    overwrite : bool

    Returns
    -------
    float (global) or SpatRaster (local).
    """
    import warnings as _warnings
    if x.nlyr() > 1:
        _warnings.warn("autocor: only the first layer of x is used")
        from .subset import subset_rast
        x = subset_rast(x, 0)

    method = method.lower()
    if method not in ("moran", "geary"):
        raise ValueError(f"method must be 'moran' or 'geary'; got {method!r}")

    if global_:
        if method == "moran":
            return _global_moran(x, w, filename, overwrite)
        else:
            return _global_geary(x, w, filename, overwrite)
    else:
        if method == "moran":
            return _local_moran(x, w, filename, overwrite)
        else:
            return _local_geary(x, w, filename, overwrite)


# ---------------------------------------------------------------------------
# layerCor — layer-wise correlation matrix
# ---------------------------------------------------------------------------

def layer_cor(
    x: SpatRaster,
    fun: str = "pearson",
    *,
    na_rm: bool = True,
    asSample: bool = True,
) -> "np.ndarray":
    """
    Compute the correlation (or covariance) matrix between layers of *x*.

    Parameters
    ----------
    x : SpatRaster
    fun : str
        ``"pearson"`` (default), ``"spearman"``, or ``"cov"``
        (covariance).
    na_rm : bool
        Ignore NA values.
    asSample : bool
        Use sample (n-1) divisor for covariance.

    Returns
    -------
    numpy.ndarray, shape (nlyr, nlyr).
    """
    nl = x.nlyr()
    vals = _read_values_layer_matrix(x)
    if na_rm:
        mask = ~np.isnan(vals).any(axis=1)
        vals = vals[mask]

    if fun == "cov":
        ddof = 1 if asSample else 0
        return np.cov(vals.T, ddof=ddof)
    elif fun == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(vals)
        if nl == 1:
            return np.array([[1.0]])
        return np.array(corr)
    else:
        return np.corrcoef(vals.T)
