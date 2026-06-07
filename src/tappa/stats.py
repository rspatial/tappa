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


def _autocor_numeric(
    x: np.ndarray, w: np.ndarray, method: str
) -> Union[float, np.ndarray]:
    """Numeric-vector overload of ``autocor`` (R ``autocor.numeric``).

    Mirrors ``R/autocor.R`` lines 46–132 for a 1-D values vector and a
    square weight matrix. Supports ``"moran"``, ``"geary"``, ``"gi"``,
    ``"gi*"``, ``"locmor"``, ``"mean"``.
    """
    import warnings as _warnings
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] != x.size:
        raise ValueError("autocor: w must be a square matrix with sides the size of x")
    if np.any(np.isnan(w)):
        raise ValueError("autocor: NA value(s) in the weight matrix")
    n = x.size
    if method in ("moran", "geary", "locmor", "gi"):
        if np.any(np.diag(w) != 0):
            _warnings.warn(
                f"autocor: it is unexpected that a weight matrix for {method} "
                "has diagonal values that are not zero"
            )
    elif method == "gi*":
        if np.any(np.diag(w) == 0):
            _warnings.warn(
                "autocor: it is unexpected that a weight matrix for gi* has "
                "diagonal values that are zero"
            )

    if method == "moran":
        dx = x - np.nanmean(x)
        pm = np.tile(dx, (n, 1)) * dx[:, None]
        return (n / np.sum(dx ** 2)) * np.sum(pm * w) / np.sum(w)
    if method == "geary":
        dx = x - np.nanmean(x)
        pm = (np.tile(dx, (n, 1)) - dx[:, None]) ** 2
        return ((n - 1) / np.sum(dx ** 2)) * np.sum(w * pm) / (2 * np.sum(w))
    if method == "locmor":
        z = x - np.nanmean(x)
        mp = z / (np.nansum(z ** 2) / n)
        return mp * np.array([np.nansum(z * w[i, :]) for i in range(n)])
    if method == "mean":
        j = np.isnan(x)
        x_ = x.copy()
        x_[j] = 0.0
        ww = w.copy()
        ww[j, :] = 0.0
        ww[:, j] = 0.0
        m = np.array([np.sum(x_ * ww[i, :]) / np.sum(ww[i, :]) for i in range(n)])
        m[j] = np.nan
        return m
    if method == "gi":
        ww = w.copy()
        np.fill_diagonal(ww, 0.0)
        sumxminx = np.nansum(x) - x
        Gi = np.sum(x[None, :] * ww, axis=1) / sumxminx
        Ei = np.sum(ww, axis=1) / (n - 1)
        xibar = sumxminx / (n - 1)
        si2 = (np.sum(x ** 2) - x ** 2) / (n - 1) - xibar ** 2
        VG = si2 * (((n - 1) * np.sum(ww ** 2, axis=1) - np.sum(ww, axis=1) ** 2) / (n - 2))
        VG = VG / sumxminx ** 2
        return (Gi - Ei) / np.sqrt(VG)
    if method == "gi*":
        Gi = np.sum(x[None, :] * w, axis=1) / np.sum(x)
        Ei = np.sum(w, axis=1) / n
        si2 = np.sum((x - x.mean()) ** 2) / n
        VG = (si2 * ((n * np.sum(w ** 2, axis=1) - np.sum(w, axis=1) ** 2) / (n - 1))) / (np.sum(x) ** 2)
        return (Gi - Ei) / np.sqrt(VG)
    raise ValueError(f"autocor: unknown method {method!r}")


def autocor(
    x: Union[SpatRaster, "np.ndarray", list, tuple],
    w: Optional[Union[str, "np.ndarray"]] = None,
    global_: bool = True,
    method: str = "moran",
    filename: str = "",
    overwrite: bool = False,
) -> Union[float, SpatRaster, "np.ndarray"]:
    """
    Compute spatial autocorrelation for *x*.

    Parameters
    ----------
    x : SpatRaster or 1-D array-like
        Single-layer raster, or a numeric vector of values (one per
        spatial unit) — same dispatch as R ``terra::autocor`` over
        ``signature(x="SpatRaster")`` and ``signature(x="numeric")``.
    w : str or array, optional
        For SpatRaster: ``"queen"``, ``"rook"`` or a 2-D weight matrix.
        For numeric vectors: a square weight matrix (required).
    global_ : bool
        If True, return a scalar (global Moran's I or Geary's C).
        If False, return a local SpatRaster.
    method : str
        ``"moran"`` (default) or ``"geary"``. The numeric overload also
        accepts ``"gi"``, ``"gi*"``, ``"locmor"`` and ``"mean"``.
    filename : str
    overwrite : bool

    Returns
    -------
    float (global) or SpatRaster (local) or numpy.ndarray (numeric overload).
    """
    method = str(method).lower()

    if not isinstance(x, SpatRaster):
        if w is None:
            raise ValueError("autocor: a weight matrix is required for vector input")
        return _autocor_numeric(np.asarray(x), np.asarray(w), method)

    import warnings as _warnings
    if x.nlyr() > 1:
        _warnings.warn("autocor: only the first layer of x is used")
        from .subset import _subset_rast
        x = _subset_rast(x, 0)

    if method not in ("moran", "geary"):
        raise ValueError(
            "autocor (SpatRaster): method must be 'moran' or 'geary'; "
            f"got {method!r}"
        )

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
