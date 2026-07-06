"""
zonal.py — zonal statistics.
"""
from __future__ import annotations
from typing import Callable, Optional, Union

from ._terra import SpatRaster, SpatOptions
from ._helpers import messages, spatoptions

_cpp_zonal = SpatRaster.zonal  # captured before any monkey-patching


def _opt() -> SpatOptions:
    return SpatOptions()


def _empty_group_raster(x: SpatRaster) -> SpatRaster:
    """
    Third argument to C++ ``zonal(z, g, ...)`` when there is no grouping layer.

    Matches R ``grast <- rast()`` passed as ``x@pntr$zonal(z@pntr, grast@pntr, ...)``.
    """
    from .rast import rast

    e = x.extent
    v = e.vector
    cr = x.get_crs("wkt")
    return rast(
        None,
        nrows=x.nrow(),
        ncols=x.ncol(),
        nlyrs=1,
        xmin=float(v[0]),
        xmax=float(v[1]),
        ymin=float(v[2]),
        ymax=float(v[3]),
        crs=cr if cr else None,
    )


_ZONAL_FUNS = {
    "sum", "mean", "median", "modal", "min", "max", "prod", "any", "all",
    "count", "sd", "std", "first", "isNA", "notNA",
}


def zonal(
    x: SpatRaster,
    z: SpatRaster,
    fun: Union[str, Callable] = "mean",
    *,
    na_rm: bool = True,
    as_raster: bool = False,
    wide: bool = True,
    filename: str = "",
    overwrite: bool = False,
) -> "pd.DataFrame":
    """
    Compute zonal statistics.

    Parameters
    ----------
    x : SpatRaster
        Values raster.
    z : SpatRaster
        Zones raster (should be integer or categorical).
    fun : str or callable
        Summary function.  Built-ins include ``"sum"``, ``"mean"``, ``"median"``,
        ``"min"``, ``"max"``, ``"sd"``, ``"isNA"`` (count NAs per zone),
        ``"notNA"`` (count non-NA values per zone), …
    na_rm : bool
        Ignore NA values.
    as_raster : bool
        If True, return a SpatRaster instead of a DataFrame.
    wide : bool
        If True (default), return one column per layer of *x*.
    filename : str
    overwrite : bool

    Returns
    -------
    pandas.DataFrame or SpatRaster
    """
    txt = fun if isinstance(fun, str) else getattr(fun, "__name__", "")
    if txt not in _ZONAL_FUNS:
        raise ValueError(f"Function {txt!r} is not supported; use one of {sorted(_ZONAL_FUNS)}")

    opt = spatoptions(filename, overwrite)
    g = _empty_group_raster(x)
    xc = _cpp_zonal(x, z, g, txt, na_rm, opt)
    result = messages(xc, "zonal")

    if as_raster:
        return result

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for zonal()")

    from ._helpers import _getSpatDF
    df = _getSpatDF(result)
    if df is None:
        # result is a SpatRaster; extract via C++ route
        nr, nc_ = result.nrow(), result.ncol()
        import numpy as np
        vals = np.array(result.readValues(0, nr, 0, nc_), dtype=float)
        from .names import _cpp_layer_names
        nms = _cpp_layer_names(result)
        df = pd.DataFrame(vals.reshape(nr * nc_, len(nms)), columns=nms)
    return df


# ---------------------------------------------------------------------------
# expanse — area covered by non-NA cells (R terra::expanse for SpatRaster)
# ---------------------------------------------------------------------------

def _replace_with_label(r: SpatRaster, df: "pd.DataFrame", col: str) -> "pd.DataFrame":
    """Replace numeric codes in ``df[col]`` with the category labels of *r* (R ``replace_with_label``)."""
    from .levels import is_factor, cats, activeCat

    ff = is_factor(r)
    if not any(ff):
        return df

    cgs = cats(r)
    df = df.copy()
    nl = len(ff)
    for f in range(nl):
        if not ff[f]:
            continue
        cg = cgs[f]
        if cg is None or len(cg.columns) < 2:
            continue
        if nl == 1:
            rows = df.index
        else:
            rows = df.index[df["layer"] == f]
        if len(rows) == 0:
            continue
        act = activeCat(r, f)
        code_col = cg.columns[0]
        lab_col = cg.columns[min(act, len(cg.columns) - 1)]
        mapping = dict(zip(cg[code_col].astype(float), cg[lab_col]))
        df[col] = df[col].astype(object)
        df.loc[rows, col] = [
            mapping.get(float(v), v) for v in df.loc[rows, col]
        ]
    return df


def expanse(
    x: SpatRaster,
    unit: str = "m",
    transform: bool = True,
    byValue: bool = False,
    zones: Optional[SpatRaster] = None,
    wide: bool = False,
    usenames: bool = False,
) -> "pd.DataFrame":
    """
    Compute the area covered by the non-NA cells of a :class:`SpatRaster`.

    Mirrors R ``terra::expanse`` for ``SpatRaster``.

    Parameters
    ----------
    x : SpatRaster
    unit : str
        ``"m"`` (square metres, default), ``"km"`` or ``"ha"``.
    transform : bool
        For planar CRSs, transform to lon/lat for accurate areas.
        Ignored (always geodesic) for lon/lat rasters.
    byValue : bool
        Report the area for each unique cell value separately.
    zones : SpatRaster, optional
        Zones raster with the same geometry as *x*; areas are reported per zone.
    wide : bool
        Reshape to wide format (one column per value or zone).
    usenames : bool
        Use layer names instead of (0-based) layer numbers in ``layer``.

    Returns
    -------
    pandas.DataFrame
        Long format columns: ``layer``, ``area``; plus ``value`` if *byValue*
        and ``zone`` if *zones* is given.
    """
    import numpy as np
    import pandas as pd

    opt = _opt()

    if zones is not None:
        if not isinstance(zones, SpatRaster):
            raise TypeError("[expanse] zones must be a SpatRaster")
        from .values import compareGeom
        compareGeom(x, zones, lyrs=False, crs=False, ext=True, rowcol=True)

        raw = x.sum_area_group(zones, unit, bool(transform), bool(byValue), opt)
        messages(x, "expanse")
        chunks = [
            np.asarray(vec, dtype=float).reshape(-1, 4)
            for vec in raw if len(vec)
        ]
        arr = np.vstack(chunks) if chunks else np.empty((0, 4))
        df = pd.DataFrame(arr, columns=["layer", "value", "zone", "area"])
        df["layer"] = df["layer"].astype(int)
        if byValue:
            df = _replace_with_label(x, df, "value")
            df = _replace_with_label(zones, df, "zone")
        else:
            df = _replace_with_label(zones, df, "zone")
            df = df.drop(columns="value")
        if wide and len(df) > 0:
            if byValue:
                df = df.pivot(index=["layer", "zone"], columns="value", values="area").fillna(0)
            else:
                df = df.pivot(index="layer", columns="zone", values="area").fillna(0)
            df.columns.name = None
            df = df.reset_index()
    else:
        raw = x.sum_area(unit, bool(transform), bool(byValue), opt)
        messages(x, "expanse")
        if byValue:
            recs = []
            for lyr, vec in enumerate(raw):
                arr = np.asarray(vec, dtype=float).reshape(-1, 2)
                for v, a in arr:
                    recs.append((lyr, v, a))
            df = pd.DataFrame(recs, columns=["layer", "value", "area"])
            df["layer"] = df["layer"].astype(int)
            df = _replace_with_label(x, df, "value")
            if wide and len(df) > 0:
                df = df.pivot(index="layer", columns="value", values="area").fillna(0)
                df.columns.name = None
                df = df.reset_index()
        else:
            v = list(raw[0]) if len(raw) else []
            df = pd.DataFrame({
                "layer": np.arange(len(v)),
                "area": np.asarray(v, dtype=float),
            })

    if usenames:
        from .names import _cpp_layer_names
        nms = _cpp_layer_names(x)
        df = df.copy()
        df["layer"] = [nms[int(i)] for i in df["layer"]]
    return df
