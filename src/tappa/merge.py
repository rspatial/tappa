"""
merge.py — merge and mosaic rasters and join vector attribute tables.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional, Union

from ._terra import SpatRaster, SpatRasterCollection, SpatVector, SpatOptions
from ._helpers import messages, spatoptions


def _opt() -> SpatOptions:
    return SpatOptions()


def _sprc_from_rasters(rasters: List[SpatRaster]) -> SpatRasterCollection:
    """Build a :class:`SpatRasterCollection` from rasters (R ``sprc(...)``)."""
    rc = SpatRasterCollection()
    for r in rasters:
        rc.add(r.deepcopy(), "")
    return rc


# ---------------------------------------------------------------------------
# Raster merge / mosaic
# ---------------------------------------------------------------------------

def merge(
    x: Union[SpatRaster, SpatVector, List[SpatRaster], List[SpatVector], tuple],
    *others: Any,
    first: bool = True,
    na_rm: bool = True,
    algo: int = 1,
    resample: bool = False,
    method: Optional[str] = None,
    filename: str = "",
    overwrite: bool = False,
    **kwargs: Any,
) -> Union[SpatRaster, SpatVector]:
    """
    Merge two or more SpatRasters that overlap or are adjacent.

    Where rasters overlap the value from *x* is used when *first=True*
    (default) or the last raster with a non-NA value when *first=False*.

    Parameters
    ----------
    x : SpatRaster
    *others : SpatRaster
        Additional rasters to merge.
    first : bool
        Use the first raster's value in overlapping areas.
    na_rm : bool
        Skip NA values when merging.
    algo : int
        Internal algorithm (1, 2, or 3).
    resample : bool
        If True, resample rasters that do not align with the first.
    method : str, optional
        Resampling method (e.g. ``"bilinear"``, ``"near"``).
        If ``None``, chosen automatically based on whether the raster
        has categorical values.
    filename : str
    overwrite : bool

    Returns
    -------
    SpatRaster
    """
    if isinstance(x, SpatVector) or (
        isinstance(x, (list, tuple)) and x and isinstance(x[0], SpatVector)
    ):
        y = others[0] if others else None
        return _merge_vect(x, y, *others[1:], **kwargs)

    if isinstance(x, (list, tuple)):
        if others:
            raise TypeError(
                "merge: when x is a list of SpatRasters, no extra positional "
                "arguments are allowed"
            )
        all_rasters = list(x)
        if not all_rasters:
            raise ValueError("merge: list of rasters is empty")
    elif not isinstance(x, SpatRaster):
        raise TypeError(
            f"merge: expected SpatRaster or SpatVector, got {type(x).__name__}"
        )
    else:
        all_rasters = [x] + list(others)
    opt = spatoptions(filename, overwrite)

    rc = _sprc_from_rasters(all_rasters)
    if method is None:
        method = ""
    xc = rc.merge(first, na_rm, algo, resample, method, opt)
    return messages(xc, "merge")


def mosaic(
    x: SpatRaster,
    *others: SpatRaster,
    fun: Union[str, Callable] = "mean",
    resample: bool = False,
    method: Optional[str] = None,
    filename: str = "",
    overwrite: bool = False,
) -> SpatRaster:
    """
    Mosaic two or more overlapping SpatRasters using an aggregation function.

    Unlike merge(), mosaic() aggregates overlapping values rather than
    giving priority to any one raster.

    Parameters
    ----------
    x : SpatRaster
    *others : SpatRaster
    fun : str or callable
        Aggregation function: ``"mean"`` (default), ``"sum"``, ``"min"``,
        ``"max"``, ``"median"``, ``"first"``, ``"last"``, or ``"blend"``
        (distance-weighted feathering that produces smooth gradients in
        overlap zones).
    resample : bool
        If True, resample rasters that do not align with the first.
    method : str, optional
        Resampling method (e.g. ``"bilinear"``, ``"near"``).
        If ``None``, chosen automatically.
    filename : str
    overwrite : bool

    Returns
    -------
    SpatRaster
    """
    fun_str = fun if isinstance(fun, str) else getattr(fun, "__name__", "mean")
    if fun_str not in {"mean", "sum", "min", "max", "median", "first", "last", "blend"}:
        raise ValueError(
            f"fun must be one of mean/sum/min/max/median/first/last/blend; got {fun_str!r}"
        )

    all_rasters = [x] + list(others)
    opt = spatoptions(filename, overwrite)
    rc = _sprc_from_rasters(all_rasters)
    if method is None:
        method = ""
    xc = rc.mosaic(fun_str, resample, method, opt)
    return messages(xc, "mosaic")


# ---------------------------------------------------------------------------
# Vector attribute table join
# ---------------------------------------------------------------------------

def _merge_vect(
    x: Union[SpatVector, List[SpatVector]],
    y: Optional[Union["pd.DataFrame", SpatVector]] = None,
    *more: SpatVector,
    **kwargs,
) -> SpatVector:
    """
    Combine SpatVectors or join an attribute table to a SpatVector.

    Two modes mirror R ``terra::merge()``:

    * **Row-bind SpatVectors** — pass a list of SpatVectors (or several
      positional arguments). Returns one SpatVector with all features.
    * **Attribute join** — pass *x* (SpatVector) and *y* (pandas.DataFrame).
      Extra ``**kwargs`` are forwarded to :func:`pandas.merge`.

    Parameters
    ----------
    x : SpatVector or list of SpatVector
    y : pandas.DataFrame or SpatVector, optional
        DataFrame to join, *or* a second SpatVector to row-bind onto *x*.
    *more : SpatVector
        Additional SpatVectors to row-bind.
    **kwargs
        Forwarded to ``pandas.merge`` for the attribute-join mode.

    Returns
    -------
    SpatVector
    """
    if isinstance(x, (list, tuple)):
        items = list(x)
        if y is not None or more:
            raise TypeError(
                "merge_vect: when x is a list of SpatVectors, no extra "
                "positional arguments are allowed"
            )
    elif isinstance(y, SpatVector) or any(isinstance(m, SpatVector) for m in more):
        items = [x] + ([y] if y is not None else []) + list(more)
    else:
        items = None  # attribute-join mode

    if items is not None:
        if not items:
            raise ValueError("merge_vect: no SpatVectors to combine")
        out = items[0].deepcopy()
        for v in items[1:]:
            out = out.append(v, True)
        return messages(out, "merge_vect")

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for merge_vect()")
    from ._helpers import _getSpatDF, _makeSpatDF

    if y is None:
        raise TypeError("merge_vect: missing argument 'y' (DataFrame or SpatVector)")

    v = _getSpatDF(x.df)
    if v is None:
        v = pd.DataFrame()
    uid_col = "__uid__"
    v[uid_col] = range(len(v))
    m = pd.merge(v, y, **kwargs)
    m = m.sort_values(uid_col).reset_index(drop=True)
    if len(m) > len(x):
        raise ValueError(
            "merge would expand the number of features; 'all.y=True' is not supported"
        )
    uid_vals = m[uid_col].dropna().astype(int).tolist()
    m = m.drop(columns=[uid_col])
    xc = x.subset_rows(uid_vals)
    sdf = _makeSpatDF(m)
    xc.set_df(sdf)
    return xc
