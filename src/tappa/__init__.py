"""
tappa — Python bindings for the terra geospatial C++ library.

The compiled extension ``_terra`` exposes the same core C++ classes as the R
package. The functions in this package mirror the **R** ``terra`` API by
name (camelCase) so that workflows translate with minimal renaming.

Two equivalent calling styles are supported:

  Functional style (R-like):        Method style (Pythonic):
  ─────────────────────────────     ─────────────────────────────────
  pt.crop(r, e)                     r.crop(e)
  pt.mask(r, mask_r)                r.mask(mask_r)
  pt.aggregate(r, 2)                r.aggregate(2)
  pt.focal(r, 3)                    r.focal(3)
  pt.values(r)                      r.values()
  pt.bufferVect(v, 1000)            v.buffer(1000)
  pt.projectVector(v, crs)          v.project(crs)

Quick reference (R → Python):
  rast()              pt.rast()
  vect()              pt.vect()
  ext()               pt.ext()
  crs()               pt.crs()
  crop()              pt.crop(r, e)  or  r.crop(e)
  mask()              pt.mask(r, m)  or  r.mask(m)
  project(rast)       pt.projectRaster()  or  r.project(crs)
  project(vect)       pt.projectVector()  or  v.project(crs)
  resample()          pt.resample()  or  r.resample(template)
  classify()          pt.classify()  or  r.classify(rcl)
  terrain()           pt.terrain()  or  r.terrain()
  focal()             pt.focal()  or  r.focal(w)
  aggregate()         pt.aggregate()  or  r.aggregate(fact)
  values()            pt.values(r)  or  r.values()
  plot()              pt.plot(r)  or  r.plot()
  spatSample()        pt.spatSample(r, n)
  setLevels()         pt.setLevels(r, df)
  ... and more in generics.py / arith.py / geom.py / methods.py

Naming conventions:
  * Public API uses **camelCase** to match terra-R (``spatSample``,
    ``setLevels``, ``compareGeom``, ``writeRaster`` …).
  * Predicates keep the Pythonic ``is_X`` / ``not_X`` snake_case form
    (``is_na``, ``is_factor``, ``is_lonlat`` …) — same as numpy/pandas.
  * A few names with the ``rast_`` prefix avoid Python builtin clashes
    (``rast_sum`` vs ``sum``, ``rast_abs`` vs ``abs`` …).
  * Trailing-underscore names dodge keywords (``round_``, ``global_``).

Arithmetic operators (Arith_generics.R):
  r + r2, r - n, r * r2, etc.   operator overloads on SpatRaster
  e + e2 (union), e * n (scale)  operator overloads on SpatExtent
  v + v2 (union), v - v2 (erase) operator overloads on SpatVector
  is_na(), not_na(), whichMax(), rast_sum(), compareRast(), …

Geometry operations (geom.R):
  is_valid(), makeValid(), unionVect(), intersectVect(), erase(),
  symdif(), bufferVect(), hull(), voronoi(), delaunay(), spin(), …
"""

from ._terra import (  # noqa: F401
    SpatRaster,
    SpatVector,
    SpatExtent,
    SpatOptions,
    SpatDataFrame,
    SpatFactor,
    SpatTime_v,
    SpatSRS,
    SpatMessages,
    SpatCategories,
    SpatVectorCollection,
    SpatVectorProxy,
    SpatRasterCollection,
    SpatRasterStack,
)

from ._helpers import characterCRS, messages                        # noqa: F401
from .crs import crs, projPipelines                                   # noqa: F401
from .show import reprExtent, reprRaster, reprVector, show, registerReprs  # noqa: F401
registerReprs()  # attach __repr__ / __str__ to C++ types
from .extent import ext                                               # noqa: F401
from .rast import rast                                                # noqa: F401
from .vect import vect                                                # noqa: F401
from .plot import plot, plotRGB, points, lines, polys, text           # noqa: F401

from .arith import (                                                  # noqa: F401
    # NA / logical tests
    is_na, not_na, is_true, is_false,
    is_nan, is_finite, is_infinite,
    any_na, all_na, no_na, count_na,
    # summaries
    whichMax, whichMin, whichLyr,
    whereMax, whereMin,
    rast_sum, rast_mean, rast_min, rast_max,
    rast_median, rast_modal, stdevRast,
    global_,
    # compare / logic
    compareRast, logicRastFn,
    # type coercion / queries
    as_int_rast, as_bool_rast,
    is_bool_rast, is_int_rast, is_num_rast,
    registerOperators,
)
registerOperators()  # attach +, -, *, /, ==, … to C++ types

from .methods import registerMethods                                 # noqa: F401
registerMethods()    # attach r.crop(), r.mask(), v.buffer(), … to C++ types

# Python-friendly conveniences on the C++ types (mirroring R `length()` /
# `nrow()`, which terra users rely on).
SpatVector.__len__ = lambda self: int(self.nrow())  # type: ignore[assignment]
SpatRaster.__len__ = lambda self: int(self.nlyr())  # type: ignore[assignment]


def _spatvector_setitem(self, key, value):
    """``v["col"] = values`` adds or replaces an attribute column.

    Mirrors R's ``v$col <- values`` / ``v[["col"]] <- values``: dispatches
    on the value's dtype to the appropriate C++ ``add_column_*`` method.
    """
    if not isinstance(key, str):
        raise TypeError("SpatVector column assignment expects a string column name")
    import numpy as _np
    arr = _np.asarray(value)
    name = str(key)
    # Replace if column already exists.
    if hasattr(self, "names") and name in list(self.names):
        try:
            self.remove_column(name)
        except Exception:
            pass
    if arr.dtype.kind in ("i", "u"):
        self.add_column_long([int(x) for x in arr.ravel()], name)
    elif arr.dtype.kind == "b":
        self.add_column_bool([int(bool(x)) for x in arr.ravel()], name)
    elif arr.dtype.kind in ("U", "S", "O"):
        self.add_column_string([str(x) for x in arr.ravel()], name)
    else:  # float, complex (cast to float)
        self.add_column_double([float(x) for x in arr.ravel()], name)


SpatVector.__setitem__ = _spatvector_setitem  # type: ignore[assignment]


def _spatvector_getitem(self, key):
    """Pythonic SpatVector indexing.

    * ``v["col"]`` — return the column values as a numpy array (R ``v$col``).
    * ``v[mask]`` (bool array, length ``nrow``) — subset features.
    * ``v[i]``  (int / numpy int) — return feature(s) at index *i*.
    * ``v[slice]`` — subset features by Python slice (0-based).
    * ``v[list[int]]`` — subset features by 0-based indices.
    """
    import numpy as _np
    from .subset import subsetVect as _subset_vect
    from ._helpers import _getSpatDF

    if isinstance(key, str):
        df = _getSpatDF(self.df)
        if df is None or key not in df.columns:
            raise KeyError(f"column {key!r} not found")
        return df[key].values

    if isinstance(key, slice):
        idx = list(range(*key.indices(self.nrow())))
        return _subset_vect(self, idx)

    if isinstance(key, (bool, _np.bool_)):
        raise TypeError("SpatVector indexing with a single bool is not supported")

    if isinstance(key, (int, _np.integer)):
        return _subset_vect(self, [int(key)])

    arr = _np.asarray(key)
    if arr.dtype == bool:
        if arr.size != self.nrow():
            raise ValueError(
                f"boolean mask length ({arr.size}) does not match nrow ({self.nrow()})"
            )
        return _subset_vect(self, arr.tolist())
    if arr.dtype.kind in ("i", "u"):
        return _subset_vect(self, arr.tolist())

    raise TypeError(f"unsupported SpatVector index type: {type(key).__name__}")


SpatVector.__getitem__ = _spatvector_getitem  # type: ignore[assignment]

from .geom import (                                                   # noqa: F401
    # validity
    is_valid, makeValid,
    # set operations
    unionVect, intersectVect, erase, symdif, coverVect,
    # crop / mask
    cropVect, maskVect,
    # geometry modifications
    bufferVect, disaggVect, flipVect, spin,
    hull, delaunay, voronoi, elongate,
    mergeLines, makeNodes, removeDupNodes,
    simplifyGeom, thinNodes, thin,
    sharedPaths, snapVect, gaps,
    forceCCW, widthVect, clearance,
    centroids,
    # predicates
    is_empty,
)

from .generics import (                                               # noqa: F401
    # dimensions / metadata
    nrow, ncol, nlyr, ncell, res, origin,
    # helpers
    spatOptions, deepcopy, tighten,
    # extent
    extAlign,
    # raster geometry
    is_rotated, is_flipped, flip, rotate, shift, rescale,
    trans, trim, revRaster,
    # raster values
    clamp, clampTS, classify, subst, cover, diffRaster,
    disagg, segregate, selectRange, sortRaster,
    rangeFill, weightedMean,
    # raster analysis
    boundaries, patches, cellSize, surfArea, terrain, shade, nidp,
    sieve, rectify, stretch, scaleLinear, scaleRaster,
    quantileRaster, atan_2,
    # raster processing
    crop, mask, projectRaster, resample, intersectRast,
    # vector
    projectVector, shiftVect, rotateVect, rescaleVect, transVect,
    # scoff
    scoff, setScoff,
    # local / cell-based
    roll, thresh, selectHighest, divide, approximate, extractRange,
)

# ---- New translation modules -----------------------------------------------
from .values import (                                                 # noqa: F401
    has_values, in_memory, sources,
    has_min_max, minMax, setMinMax,
    values, setValues, focalValues,
    vectValues, setVectValues,
    compareGeom,
)
from .levels import (                                                 # noqa: F401
    is_factor, asFactor,
    levels, setLevels,
    cats, setCats, categories,
    activeCat, setActiveCat,
    addCats, dropLevels, concats, catalyze,
    has_colors, coltab, setColtab,
)
from .names import (                                                  # noqa: F401
    namesRast, setNamesRast, setNamesInplace,
    namesVect, setNamesVect,
    varnames, setVarnames,
    longnames, setLongnames,
)
from .app import app, lapp, tapp, xapp, rapp, sapp                   # noqa: F401
from .focal import focal, focal3D, focalMat                          # noqa: F401
from .aggregate import aggregate, disagg as aggregateDisagg, aggregateVect  # noqa: F401
from .zonal import zonal                                              # noqa: F401
from .crosstab import crosstab                                        # noqa: F401
from .freq import freq                                                # noqa: F401
from .flowAccumulation import flowAccumulation                     # noqa: F401
from .pitfinder import pitfinder                                     # noqa: F401
from .extract import extract, extractXY                              # noqa: F401
from .math import (                                                   # noqa: F401
    math, log, sqrt, abs_ as rast_abs, ceiling, floor,
    round_, cumsum, cumprod, cummax, cummin,
    floorExt, ceilingExt, roundExt,
    ifel,
)
from .cells import (                                                  # noqa: F401
    cells,
    rowFromY, colFromX,
    cellFromXY, cellFromRowCol,
    xyFromCell, rowColFromCell,
)
from .init import init                                                # noqa: F401
from .distance import (                                               # noqa: F401
    bufferRast, distanceRast,
    costDist, gridDist,
    distanceXY, distanceVectSelf, distanceVect, distancePoints,
)
from .rasterize import rasterize, rasterize_geom                     # noqa: F401
from .time import has_time, timeInfo, getTime, setTime            # noqa: F401
from .write import (                                                  # noqa: F401
    writeRaster, writeStart, writeValues, writeStop, blocks,
    writeVector, update,
)
from .sample import spatSample, gridSample                         # noqa: F401
from .stats import (                                                  # noqa: F401
    rowSums, colSums, rowMeans, colMeans,
    matchRast, is_in,
    autocor, layerCor,
)
from .merge import merge as mergeRast, mosaic, mergeVect            # noqa: F401
from .relate import is_related, relate, relateSelf, adjacent, nearby  # noqa: F401
from .subset import subsetRast, subsetVect                         # noqa: F401
from .window import has_window, setWindow, removeWindow, extend    # noqa: F401
from .coerce import (                                                 # noqa: F401
    asPolygons, asLines, asPoints,
    asArray, asMatrix, asDataFrame,
)
from .spatvec import (                                                # noqa: F401
    geomtype, is_lines, is_polygons, is_points,
    geom, crds,
    expanse, perim, nseg,
    fillHoles, vectAsDF, geomAsWkt,
)
from .sds import SpatRasterDataset, sds                              # noqa: F401
from .sprc import SprcCollection, sprc                               # noqa: F401
from .tessellate import tessellate                                   # noqa: F401
from .tileApply import tileApply, getTileExtents, makeTiles    # noqa: F401

__version__ = "0.1.0"

__all__ = [
    # High-level API (R-like)
    "rast", "vect", "ext", "crs", "projPipelines",
    "registerMethods",
    "plot", "plotRGB", "points", "lines", "polys", "text",
    "messages", "characterCRS",
    "show", "reprRaster", "reprVector", "reprExtent",
    # values
    "has_values", "in_memory", "sources",
    "has_min_max", "minMax", "setMinMax",
    "values", "setValues", "focalValues",
    "vectValues", "setVectValues", "compareGeom",
    # levels / colors
    "is_factor", "asFactor",
    "levels", "setLevels",
    "cats", "setCats", "categories",
    "activeCat", "setActiveCat",
    "addCats", "dropLevels", "concats", "catalyze",
    "has_colors", "coltab", "setColtab",
    # names
    "namesRast", "setNamesRast", "setNamesInplace",
    "namesVect", "setNamesVect",
    "varnames", "setVarnames",
    "longnames", "setLongnames",
    # app
    "app", "lapp", "tapp", "xapp", "rapp", "sapp",
    # focal
    "focal", "focal3D", "focalMat",
    # aggregate
    "aggregate", "aggregateDisagg", "aggregateVect",
    # zonal
    "zonal",
    # crosstab
    "crosstab",
    # freq
    "freq",
    # flow accumulation
    "flowAccumulation",
    # pitfinder
    "pitfinder",
    # extract
    "extract", "extractXY",
    # math
    "math", "log", "sqrt", "rast_abs", "ceiling", "floor",
    "round_", "cumsum", "cumprod", "cummax", "cummin",
    "floorExt", "ceilingExt", "roundExt",
    "ifel",
    # cells
    "cells",
    "rowFromY", "colFromX",
    "cellFromXY", "cellFromRowCol",
    "xyFromCell", "rowColFromCell",
    # init
    "init",
    # distance
    "bufferRast", "distanceRast",
    "costDist", "gridDist",
    "distanceXY", "distanceVectSelf", "distanceVect", "distancePoints",
    # rasterize
    "rasterize", "rasterize_geom",
    # time
    "has_time", "timeInfo", "getTime", "setTime",
    # write
    "writeRaster", "writeStart", "writeValues", "writeStop", "blocks",
    "writeVector", "update",
    # sample
    "spatSample", "gridSample",
    # stats
    "rowSums", "colSums", "rowMeans", "colMeans",
    "matchRast", "is_in",
    "autocor", "layerCor",
    # merge
    "mergeRast", "mosaic", "mergeVect",
    # relate
    "is_related", "relate", "relateSelf",
    # subset
    "subsetRast", "subsetVect",
    # window
    "has_window", "setWindow", "removeWindow", "extend",
    # coerce
    "asPolygons", "asLines", "asPoints",
    "asArray", "asMatrix", "asDataFrame",
    # spatvec
    "geomtype", "is_lines", "is_polygons", "is_points",
    "geom", "crds",
    "expanse", "perim", "nseg",
    "fillHoles", "vectAsDF", "geomAsWkt",
    # dimensions
    "nrow", "ncol", "nlyr", "ncell", "res", "origin",
    # helpers
    "spatOptions", "deepcopy", "tighten",
    # extent
    "extAlign",
    # raster geometry
    "is_rotated", "is_flipped", "flip", "rotate", "shift", "rescale",
    "trans", "trim", "revRaster",
    # raster values
    "clamp", "clampTS", "classify", "subst", "cover", "diffRaster",
    "disagg", "segregate", "selectRange", "sortRaster",
    "rangeFill", "weightedMean",
    # raster analysis
    "boundaries", "patches", "cellSize", "surfArea", "terrain", "shade", "nidp",
    "sieve", "rectify", "stretch", "scaleLinear", "scaleRaster",
    "quantileRaster", "atan_2",
    # raster processing
    "crop", "mask", "projectRaster", "resample", "intersectRast",
    # vector
    "projectVector", "shiftVect", "rotateVect", "rescaleVect", "transVect",
    # scoff
    "scoff", "setScoff",
    "roll", "thresh", "selectHighest", "divide", "approximate", "extractRange",
    # arith (Arith_generics.R)
    "is_na", "not_na", "is_true", "is_false",
    "is_nan", "is_finite", "is_infinite",
    "any_na", "all_na", "no_na", "count_na",
    "whichMax", "whichMin", "whichLyr",
    "whereMax", "whereMin",
    "rast_sum", "rast_mean", "rast_min", "rast_max", "global_",
    "rast_median", "rast_modal", "stdevRast",
    "compareRast", "logicRastFn",
    "as_int_rast", "as_bool_rast",
    "is_bool_rast", "is_int_rast", "is_num_rast",
    # geom (geom.R)
    "is_valid", "makeValid",
    "unionVect", "intersectVect", "erase", "symdif", "coverVect",
    "cropVect", "maskVect",
    "bufferVect", "disaggVect", "flipVect", "spin",
    "hull", "delaunay", "voronoi", "elongate",
    "mergeLines", "makeNodes", "removeDupNodes",
    "simplifyGeom", "thinNodes", "thin",
    "sharedPaths", "snapVect", "gaps",
    "forceCCW", "widthVect", "clearance",
    "is_empty",
    # Core types (C++)
    "SpatRaster", "SpatVector", "SpatExtent", "SpatOptions",
    "SpatDataFrame", "SpatFactor", "SpatTime_v", "SpatSRS",
    "SpatMessages", "SpatCategories", "SpatVectorCollection",
    "SpatVectorProxy", "SpatRasterCollection", "SpatRasterStack",
    # sds / sprc
    "SpatRasterDataset", "sds",
    "SprcCollection", "sprc",
    # tessellate
    "tessellate",
    # tileApply
    "tileApply", "getTileExtents", "makeTiles",
    "__version__",
]
