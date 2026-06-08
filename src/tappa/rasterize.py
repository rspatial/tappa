"""
rasterize.py — convert vector data to raster.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional, Union
import numpy as np

from ._terra import SpatRaster, SpatVector, SpatOptions
from ._helpers import messages, spatoptions
from .names import _cpp_layer_names


def _opt() -> SpatOptions:
    return SpatOptions()


_RASTERIZE_FUNS = {"first", "last", "pa", "sum", "mean", "count", "min", "max", "prod"}


def _normalize_rasterize_fun(fun: Optional[Union[str, Callable]]) -> str:
    if fun is None:
        return "last"
    if isinstance(fun, str):
        return fun.lower()
    return getattr(fun, "__name__", "last")


def _is_na_scalar(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val == "NA":
        return True
    try:
        import pandas as pd
        if pd.isna(val):
            return True
    except ImportError:
        pass
    return False


def _point_rasterize_values(
    x: SpatVector,
    field: Union[str, float, int, None],
    fun: str,
    n: int,
) -> np.ndarray:
    """
    Build per-point values for ``rasterizePointsXY``, mirroring R
    ``rasterize_points()`` / ``rasterize(SpatVector, ...)`` for points.

    For ``fun='count'``, values are only used to honour ``na_rm`` in the C++
    core (``std::isnan``); non-NA points are counted regardless of field
    content, matching terra.
    """
    if isinstance(field, (int, float)) and not isinstance(field, bool):
        return np.full(n, float(field))

    if not (isinstance(field, str) and field != ""):
        return np.ones(n, dtype=float)

    if field not in _cpp_layer_names(x):
        raise ValueError(f"{field!r} is not a field in x")

    import pandas as pd
    from ._helpers import _getSpatDF

    df = _getSpatDF(x.df)
    if df is None or field not in df.columns:
        raise ValueError(f"{field!r} is not a field in x")

    col = df[field]
    # R ``$`` on a data.frame with duplicate names returns the first match.
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]

    if fun == "count":
        vals = np.ones(len(col), dtype=float)
        for i, v in enumerate(col):
            if _is_na_scalar(v):
                vals[i] = np.nan
        return vals

    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float)

    if pd.api.types.is_bool_dtype(col):
        return col.to_numpy(dtype=float)

    # Character / factor columns: terra coerces to 0-based integer codes.
    codes, _ = pd.factorize(col, use_na_sentinel=True)
    vals = codes.astype(float)
    vals[codes < 0] = np.nan
    return vals


def rasterize(
    x: Union[SpatVector, np.ndarray, "pd.DataFrame"],
    y: SpatRaster,
    field: Union[str, float, int, None] = "",
    fun: Optional[Union[str, Callable]] = None,
    *,
    background: float = float("nan"),
    touches: bool = False,
    update: bool = False,
    cover: bool = False,
    by: Optional[str] = None,
    na_rm: bool = False,
    filename: str = "",
    overwrite: bool = False,
) -> SpatRaster:
    """
    Convert a SpatVector (or coordinate matrix) to a SpatRaster.

    Parameters
    ----------
    x : SpatVector, ndarray (n×2 point coords), or DataFrame
        Features to rasterize.
    y : SpatRaster
        Template raster (extent, resolution, CRS).
    field : str, float, or None
        Attribute column to use as cell values, a numeric constant, or
        ``""`` (presence = 1).
    fun : str or callable, optional
        Aggregation function for overlapping cells.  Built-ins: ``"first"``,
        ``"last"``, ``"sum"``, ``"mean"``, ``"count"``, ``"min"``, ``"max"``,
        ``"pa"`` (presence/absence).
    background : float
        Value for cells not covered by any feature.
    touches : bool
        Include cells touched by polygon boundaries.
    update : bool
        Update (fill) *y* with rasterized values rather than creating a new
        blank raster.
    cover : bool
        Return fractional cell coverage rather than feature values.
    by : str, optional
        Split *x* by the values of this column and rasterize separately
        (returns a multi-layer raster).
    na_rm : bool
        Ignore NA values when aggregating.
    filename : str
    overwrite : bool

    Returns
    -------
    SpatRaster
    """
    import pandas as pd

    if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        x_crds = arr[:, :2]
        values_arr = np.ones(len(x_crds), dtype=float)
        return _rasterize_points_xy(x_crds, values_arr, y, fun, background, update, na_rm, filename, overwrite)

    if not isinstance(x, SpatVector):
        raise TypeError("x must be a SpatVector, ndarray, or DataFrame")

    if by is not None:
        svc = messages(x.split(by), "split")
        parts = [svc.get(i) for i in range(svc.size())]
        layers = [
            rasterize(
                p,
                y,
                field=field,
                fun=fun,
                background=background,
                touches=touches,
                update=update,
                cover=cover,
                na_rm=na_rm,
            )
            for p in parts
        ]
        if not layers:
            raise ValueError("rasterize(by=...): split produced no groups")
        opt = SpatOptions()
        out = layers[0].deepcopy()
        for lr in layers[1:]:
            out.addSource(lr.deepcopy(), True, opt)
        out = messages(out, "rast")
        split_names = _cpp_layer_names(svc)
        if split_names and len(split_names) == out.nlyr():
            from .names import _set_names_rast

            out = _set_names_rast(out, [str(n) for n in split_names])
        return out

    geom_type_raw = x.geomtype()
    geom_type = geom_type_raw[0] if isinstance(geom_type_raw, list) else geom_type_raw

    if "points" in geom_type.lower():
        xy = np.array(x.coordinates(), dtype=float).reshape(-1, 2)
        fun_str = _normalize_rasterize_fun(fun)
        values_arr = _point_rasterize_values(x, field, fun_str, len(xy))
        return _rasterize_points_xy(
            xy, values_arr, y, fun_str, background, update, na_rm, filename, overwrite
        )

    # Lines / polygons via C++ SpatRaster::rasterize(x, field, values, background,
    # touches, fun, weights, update, minmax, opt) — see R/rasterize.R
    values_vec: List[float] = [1.0]
    field_str = ""
    fun_str = ""

    if cover and "polygons" in geom_type.lower():
        # R: rasterize(..., "", 1, background, touches, "", TRUE, FALSE, TRUE, opt)
        pass
    else:
        if isinstance(field, (int, float)) and not isinstance(field, bool):
            values_vec = [float(field)]
        elif isinstance(field, str) and field != "":
            if field not in _cpp_layer_names(x):
                raise ValueError(f"{field!r} is not a field in x")
            field_str = field
            if na_rm:
                from ._helpers import _getSpatDF

                df = _getSpatDF(x.df)
                col = df[field_str]
                # Only numeric columns can carry NA in a meaningful sense
                # for rasterize; for non-numeric leave the rows alone.
                try:
                    col_f = np.asarray(col, dtype=float)
                    mask = ~np.isnan(col_f)
                except (TypeError, ValueError):
                    mask = np.ones(len(col), dtype=bool)
                if not mask.all():
                    from .subset import _subset_vect

                    x = _subset_vect(x, mask.tolist())

        if fun is not None:
            fun_str = fun if isinstance(fun, str) else getattr(fun, "__name__", "last")

    opt = spatoptions(filename, overwrite)
    xc = y.rasterize(
        x,
        field_str,
        values_vec,
        float(background),
        touches,
        fun_str,
        cover,
        update,
        True,
        opt,
    )
    return messages(xc, "rasterize")


def _rasterize_points_xy(
    xy: np.ndarray,
    values: np.ndarray,
    template: SpatRaster,
    fun: Optional[Union[str, Callable]],
    background: float,
    update: bool,
    na_rm: bool,
    filename: str,
    overwrite: bool,
) -> SpatRaster:
    fun_str = "last"
    if fun is not None:
        fun_str = fun if isinstance(fun, str) else getattr(fun, "__name__", "last")
    if fun_str not in _RASTERIZE_FUNS:
        fun_str = "last"
    opt = spatoptions(filename if not update else "", True if not update else overwrite)
    xc = template.rasterizePointsXY(
        xy[:, 0].tolist(), xy[:, 1].tolist(),
        fun_str, values.tolist(), na_rm, background, opt
    )
    result = messages(xc, "rasterize")
    if update:
        from .generics import cover as rast_cover
        result = rast_cover(result, template, filename=filename, overwrite=overwrite)
    return result


# ---------------------------------------------------------------------------
# rasterizeGeom
# ---------------------------------------------------------------------------

def rasterize_geom(
    x: SpatVector,
    y: SpatRaster,
    fun: str = "count",
    unit: str = "m",
    filename: str = "",
    overwrite: bool = False,
) -> SpatRaster:
    """
    Rasterize a geometric property (area, length, or count) of *x*.

    Parameters
    ----------
    x : SpatVector
    y : SpatRaster
        Template raster.
    fun : str
        ``"count"`` (default), ``"area"``, or ``"length"``.
    unit : str
        Units for area/length: ``"m"`` or ``"km"``.
    filename : str
    overwrite : bool

    Returns
    -------
    SpatRaster
    """
    opt = spatoptions(filename, overwrite)
    xc = y.rasterizeGeom(x, unit, fun, opt)
    return messages(xc, "rasterizeGeom")
