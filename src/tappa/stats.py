"""
stats.py — row/column statistics and value matching for SpatRaster.

Covers functionality from rowSums.R, match.R, autocor.R, layerCor.R.
"""
from __future__ import annotations
from typing import Any, List, Optional, Sequence, Union
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
# layerCor — layer-wise correlation / covariance matrix
# ---------------------------------------------------------------------------

_USE_OPTS = (
    "everything",
    "complete.obs",
    "pairwise.complete.obs",
    "masked.complete",
)


def _layer_cor_cpp_use(use: str) -> str:
    return {
        "everything": "all.observations",
        "complete.obs": "complete.observations",
        "pairwise.complete.obs": "pairwise.complete.observations",
        "masked.complete": "complete.observations",
    }[use]


def _layer_cor_flat_matrix(
    flat: Sequence[float], nl: int, lyr_names: List[str]
) -> np.ndarray:
    """Reshape C++ ``layerCor`` output (R ``matrix(..., byrow=TRUE)``)."""
    mat = np.array(flat, dtype=float).reshape(nl, nl, order="C")
    return mat


def _layer_cor_apply_fun(
    fun, a: np.ndarray, b: np.ndarray, **kwargs
) -> float:
    result = fun(a, b, **kwargs)
    arr = np.asarray(result, dtype=float)
    if arr.ndim == 2:
        return float(arr[0, 1])
    return float(arr)


def _layer_cor_callable(
    x: SpatRaster,
    fun,
    *,
    use: str,
    na_rm: bool,
    maxcell: float,
    lyr_names: List[str],
    **kwargs,
) -> np.ndarray:
    from .generics import ncell
    from .sample import spatSample

    nl = x.nlyr()
    size = ncell(x) if not np.isfinite(maxcell) else int(maxcell)
    v = spatSample(x, size=size, method="regular", na_rm=na_rm, warn=False)
    cols = [c for c in v.columns if c in lyr_names]
    if len(cols) != nl:
        cols = list(v.columns[:nl])

    mat = np.full((nl, nl), np.nan, dtype=float)
    for i in range(nl):
        for j in range(i, nl):
            ci, cj = cols[i], cols[j]
            if use == "pairwise.complete.obs":
                pair = v[[ci, cj]].dropna()
                val = _layer_cor_apply_fun(
                    fun, pair[ci].to_numpy(), pair[cj].to_numpy(), **kwargs
                )
            else:
                val = _layer_cor_apply_fun(
                    fun, v[ci].to_numpy(), v[cj].to_numpy(), **kwargs
                )
            mat[i, j] = mat[j, i] = val
    return mat


def _layer_cor_cov(
    x: SpatRaster,
    *,
    use: str,
    na_rm: bool,
    asSample: bool,
    lyr_names: List[str],
) -> dict:
    from .arith import any_na
    from .generics import mask, ncell

    nl = x.nlyr()
    n = ncell(x)
    if use == "complete.obs":
        x = mask(x, any_na(x))

    vals = _read_values_layer_matrix(x)
    means = np.full((nl, nl), np.nan, dtype=float)
    cov = np.full((nl, nl), np.nan, dtype=float)
    nn = np.full((nl, nl), np.nan, dtype=float)

    for i in range(nl):
        for j in range(i, nl):
            vi = vals[:, i]
            vj = vals[:, j]
            if use == "pairwise.complete.obs":
                ok = np.isfinite(vi) & np.isfinite(vj)
                vi = vi[ok]
                vj = vj[ok]
                n_ij = vi.size
            else:
                n_ij = n
            avg_i = float(np.nanmean(vi)) if na_rm else float(np.mean(vi))
            avg_j = float(np.nanmean(vj)) if na_rm else float(np.mean(vj))
            prod = (vi - avg_i) * (vj - avg_j)
            if na_rm:
                prod = prod[np.isfinite(prod)]
            denom = n_ij - int(asSample)
            v = float(np.sum(prod)) / denom if denom else float("nan")
            cov[i, j] = cov[j, i] = v
            means[i, j] = avg_i
            means[j, i] = avg_j
            nn[i, j] = nn[j, i] = n_ij

    return {
        "covariance": cov,
        "mean": means,
        "n": nn,
    }


def layerCor(
    x: SpatRaster,
    fun: Union[str, Any] = "cor",
    w: Optional[SpatRaster] = None,
    *,
    asSample: bool = True,
    use: str = "everything",
    maxcell: float = float("inf"),
    na_rm: Optional[bool] = None,
    **kwargs: Any,
) -> Union[dict, np.ndarray]:
    """
    Correlation or covariance between layers of a :class:`SpatRaster`.

    Mirrors R ``terra::layerCor``.

    Parameters
    ----------
    x : SpatRaster
    fun : str or callable
        ``"cor"`` (Pearson), ``"cov"``, ``"weighted.cov"``, or a function
        taking two numeric vectors (e.g. ``numpy.corrcoef``).
    w : SpatRaster, optional
        Weights for ``"weighted.cov"``.
    asSample : bool
        Use sample (``n-1``) divisor for covariance.
    use : str
        NA policy: ``"everything"``, ``"complete.obs"``,
        ``"pairwise.complete.obs"``, or ``"masked.complete"``.
    maxcell : int
        Maximum number of cells to use (regular sample if smaller than
        ``ncell(x)``).
    na_rm : bool, optional
        Deprecated; ``na_rm=True`` with ``use="everything"`` selects
        pairwise complete observations.

    Returns
    -------
    dict or numpy.ndarray
        For ``fun="cor"``: ``{"correlation", "mean", "n"}``.
        For ``fun="cov"``: ``{"covariance", "mean", "n"}``.
        For a callable: correlation matrix as a 2-D array.
    """
    from .generics import ncell
    from .names import names as layer_names
    from .sample import spatSample

    if use not in _USE_OPTS:
        raise ValueError(
            f"layerCor: use must be one of {_USE_OPTS}; got {use!r}"
        )
    if na_rm is True and use == "everything":
        use = "pairwise.complete.obs"

    na_rm_flag = use != "everything"
    nl = x.nlyr()
    if nl < 2:
        raise ValueError("layerCor: x must have at least 2 layers")

    lyr_names = layer_names(x)
    callable_fun = None
    fun_name = ""

    if isinstance(fun, str):
        fun_name = fun.lower()
        if fun_name == "pearson":
            fun_name = "cor"
        if fun_name not in ("cor", "cov", "weighted.cov"):
            raise ValueError(
                "layerCor: character fun must be one of "
                "'cor', 'cov', or 'weighted.cov'"
            )
        if fun_name == "weighted.cov":
            if w is None:
                raise ValueError(
                    "layerCor: weighted.cov requires a weights layer (w)"
                )
            if w.nlyr() != 1:
                raise ValueError("layerCor: weights must be a single layer")
    else:
        callable_fun = fun
        fun_name = ""

    if np.isfinite(maxcell) and maxcell < ncell(x):
        x = spatSample(x, size=int(maxcell), method="regular", as_raster=True)

    if fun_name == "cor":
        opt = _opt()
        raw = x.layerCor("cor", _layer_cor_cpp_use(use), bool(asSample), opt)
        x = messages(x, "layerCor")
        return {
            "correlation": _layer_cor_flat_matrix(raw[0], nl, lyr_names),
            "mean": _layer_cor_flat_matrix(raw[1], nl, lyr_names),
            "n": _layer_cor_flat_matrix(raw[2], nl, lyr_names),
        }

    if fun_name == "cov":
        return _layer_cor_cov(
            x, use=use, na_rm=na_rm_flag, asSample=asSample, lyr_names=lyr_names
        )

    if fun_name == "weighted.cov":
        raise NotImplementedError("layerCor: weighted.cov is not yet implemented")

    return _layer_cor_callable(
        x,
        callable_fun,
        use=use,
        na_rm=na_rm_flag,
        maxcell=maxcell,
        lyr_names=lyr_names,
        **kwargs,
    )


layer_cor = layerCor
