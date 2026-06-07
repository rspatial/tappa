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
  pt.buffer(v, 1000)                v.buffer(1000)
  pt.project(v, crs)                v.project(crs)

Quick reference (R → Python):
  rast()              pt.rast()
  vect()              pt.vect()
  ext()               pt.ext()
  crs()               pt.crs()
  crop()              pt.crop(x, e)  or  x.crop(e)   (raster or vector)
  mask()              pt.mask(x, m)  or  x.mask(m)   (raster or vector)
  buffer()            pt.buffer(x, w)  or  x.buffer(w)
  project()           pt.project(x, crs)  or  x.project(crs)
  intersect()         pt.intersect(x, y)  or  x.intersect(y)
  distance()          pt.distance(x, y)  or  x.distance(y)
  write()             pt.write(x, file)  or  x.write(file)
  merge()             pt.merge(x, y)  or  x.merge(y)
  subset()            pt.subset(x, i)  or  x[[i]]
  names()             pt.names(x)  or  x.names()
  values()            pt.values(x)  or  x.values()
  aggregate()         pt.aggregate(x, …)  or  x.aggregate(…)
  flip/shift/…        pt.flip(x)  pt.shift(x)  pt.disagg(x)  …
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
  is_valid(), make_valid(), union(), intersect(), erase(),
  symdif(), buffer(), hull(), voronoi(), delaunay(), spin(), …
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

from ._helpers import characterCRS as character_crs, messages        # noqa: F401
from .crs import crs, projPipelines as proj_pipelines                # noqa: F401
from .show import (                                                  # noqa: F401
    reprExtent as repr_extent,
    reprRaster as repr_raster,
    reprVector as repr_vector,
    show,
    registerReprs as register_reprs,
)
register_reprs()  # attach __repr__ / __str__ to C++ types
from .extent import ext                                               # noqa: F401
from .rast import rast                                                # noqa: F401
from .vect import vect                                                # noqa: F401
from .plot import plot, plot_rgb, points, lines, polys, text           # noqa: F401

from .arith import (                                                  # noqa: F401
    # NA / logical tests
    is_na, not_na, is_true, is_false,
    is_nan, is_finite, is_infinite,
    any_na, all_na, no_na, count_na,
    # summaries
    which_max, which_min, which_lyr,
    where_max, where_min,
    rast_sum, rast_mean, rast_min, rast_max,
    rast_median, rast_modal, stdev_rast,
    global_,
    # compare / logic
    compare_rast, logic_rast_fn,
    # type coercion / queries
    as_int_rast, as_bool_rast,
    is_bool_rast, is_int_rast, is_num_rast,
    register_operators,
)
register_operators()  # attach +, -, *, /, ==, … to C++ types

from .methods import register_methods                                 # noqa: F401
register_methods()    # attach r.crop(), r.mask(), v.buffer(), … to C++ types

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
    from .names import _cpp_layer_names
    if name in _cpp_layer_names(self):
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
    from .subset import _subset_vect
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
    is_valid, make_valid,
    # set operations
    union, erase, symdif,
    # geometry modifications
    spin,
    hull, delaunay, voronoi, elongate,
    merge_lines, make_nodes, remove_dup_nodes,
    simplify_geom, thin_nodes, thin,
    shared_paths, snap, gaps,
    force_ccw, width, clearance,
    centroids,
    # predicates
    is_empty,
)

from .dispatch import buffer, project, intersect                      # noqa: F401

from .generics import (                                               # noqa: F401
    # dimensions / metadata
    nrow, ncol, nlyr, ncell, res, origin,
    # helpers
    spat_options, deepcopy, tighten,
    # extent
    ext_align,
    # raster geometry
    is_rotated, is_flipped, flip, rotate, shift, rescale,
    trans, trim, rev_raster,
    # raster values
    clamp, clamp_ts, classify, subst, cover, diff_raster,
    disagg, segregate, selectRange, sort_raster,
    range_fill, weighted_mean,
    # raster analysis
    boundaries, patches, cellSize, surfArea, terrain, shade, nidp,
    sieve, rectify, stretch, scale_linear, scale_raster,
    quantile_raster, atan_2,
    # raster processing
    crop, mask, resample,
    # scoff
    scoff, scoff_set,
    # local / cell-based
    roll, thresh, select_highest, divide, approximate, extract_range,
)

# ---- New translation modules -----------------------------------------------
from .values import (                                                 # noqa: F401
    has_values, in_memory, sources,
    has_min_max, minMax as min_max, setMinMax as set_min_max,
    values, set_values, setValues, focalValues as focal_values,
    compareGeom as compare_geom,
)
from .levels import (                                                 # noqa: F401
    is_factor, asFactor as as_factor,
    levels, setLevels as set_levels,
    cats, setCats as set_cats, categories,
    activeCat as active_cat, setActiveCat as set_active_cat,
    addCats as add_cats, dropLevels as drop_levels, concats, catalyze,
    has_colors, coltab, setColtab as set_coltab,
)
from .names import (                                                  # noqa: F401
    names, set_names,
    varnames, setVarnames as set_varnames,
    longnames, setLongnames as set_longnames,
)
from .app import app, lapp, tapp, xapp, rapp, sapp                   # noqa: F401
from .focal import focal, focal3D, focalMat as focal_mat              # noqa: F401
from .aggregate import aggregate, disagg as aggregate_disagg          # noqa: F401
from .zonal import zonal                                              # noqa: F401
from .crosstab import crosstab                                        # noqa: F401
from .freq import freq                                                # noqa: F401
from .flowAccumulation import flowAccumulation as flow_accumulation   # noqa: F401
from .pitfinder import pitfinder                                     # noqa: F401
from .extract import extract, extract_xy                             # noqa: F401
from .math import (                                                   # noqa: F401
    math, log, sqrt, abs_ as rast_abs, ceiling, floor,
    round_, cumsum, cumprod, cummax, cummin,
    floorExt as floor_ext, ceilingExt as ceiling_ext, roundExt as round_ext,
    ifel,
)
from .cells import (                                                  # noqa: F401
    cells,
    rowFromY as row_from_y, colFromX as col_from_x,
    cellFromXY as cell_from_xy, cellFromRowCol as cell_from_row_col,
    xyFromCell as xy_from_cell, rowColFromCell as row_col_from_cell,
)
from .init import init                                                # noqa: F401
from .distance import (                                               # noqa: F401
    distance,
    costDist as cost_dist, gridDist as grid_dist,
    distanceXY as distance_xy, distancePoints as distance_points,
)
from .rasterize import rasterize, rasterize_geom                     # noqa: F401
from .time import has_time, timeInfo as time_info, getTime as get_time, setTime as set_time  # noqa: F401
from .write import (                                                  # noqa: F401
    write, write_start, write_values, write_stop, blocks,
    update,
)
from .sample import spatSample as spat_sample, gridSample as grid_sample  # noqa: F401
from .stats import (                                                  # noqa: F401
    row_sums, col_sums, row_means, col_means,
    match_rast, is_in,
    autocor, layer_cor,
)
from .merge import merge, mosaic                                      # noqa: F401
from .relate import is_related, relate, relate_self, adjacent, nearby  # noqa: F401
from .subset import subset                                            # noqa: F401
from .window import has_window, setWindow as set_window, removeWindow as remove_window, extend  # noqa: F401
from .coerce import (                                                 # noqa: F401
    asPolygons as as_polygons, asLines as as_lines, asPoints as as_points,
    asArray as as_array, asMatrix as as_matrix, asDataFrame as as_data_frame,
)
from .spatvec import (                                                # noqa: F401
    geomtype, is_lines, is_polygons, is_points,
    geom, crds,
    expanse, perim, nseg,
    fillHoles as fill_holes, vectAsDF as vect_as_df, geomAsWkt as geom_as_wkt,
)
from .sds import SpatRasterDataset, sds                              # noqa: F401
from .sprc import SprcCollection, sprc                               # noqa: F401
from .tessellate import tessellate                                   # noqa: F401
from .tileApply import tileApply as tile_apply, getTileExtents as get_tile_extents, makeTiles as make_tiles  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    # High-level API (R-like)
    "rast", "vect", "ext", "crs", "proj_pipelines",
    "register_methods",
    "plot", "plot_rgb", "points", "lines", "polys", "text",
    "messages", "character_crs",
    "show", "repr_raster", "repr_vector", "repr_extent",
    # values
    "has_values", "in_memory", "sources",
    "has_min_max", "min_max", "set_min_max",
    "values", "set_values", "setValues", "focal_values",
    "compare_geom",
    # levels / colors
    "is_factor", "as_factor",
    "levels", "set_levels",
    "cats", "set_cats", "categories",
    "active_cat", "set_active_cat",
    "add_cats", "drop_levels", "concats", "catalyze",
    "has_colors", "coltab", "set_coltab",
    # names
    "names", "set_names",
    "varnames", "set_varnames",
    "longnames", "set_longnames",
    # app
    "app", "lapp", "tapp", "xapp", "rapp", "sapp",
    # focal
    "focal", "focal3D", "focal_mat",
    # aggregate
    "aggregate",
    # zonal
    "zonal",
    # crosstab
    "crosstab",
    # freq
    "freq",
    # flow accumulation
    "flow_accumulation",
    # pitfinder
    "pitfinder",
    # extract
    "extract", "extract_xy",
    # math
    "math", "log", "sqrt", "rast_abs", "ceiling", "floor",
    "round_", "cumsum", "cumprod", "cummax", "cummin",
    "floor_ext", "ceiling_ext", "round_ext",
    "ifel",
    # cells
    "cells",
    "row_from_y", "col_from_x",
    "cell_from_xy", "cell_from_row_col",
    "xy_from_cell", "row_col_from_cell",
    # init
    "init",
    # distance
    "distance",
    "cost_dist", "grid_dist",
    "distance_xy", "distance_points",
    # rasterize
    "rasterize", "rasterize_geom",
    # time
    "has_time", "time_info", "get_time", "set_time",
    # write
    "write", "write_start", "write_values", "write_stop", "blocks",
    "update",
    # sample
    "spat_sample", "grid_sample",
    # stats
    "row_sums", "col_sums", "row_means", "col_means",
    "match_rast", "is_in",
    "autocor", "layer_cor",
    # merge
    "merge", "mosaic",
    # relate
    "is_related", "relate", "relate_self",
    # subset
    "subset",
    # window
    "has_window", "set_window", "remove_window", "extend",
    # coerce
    "as_polygons", "as_lines", "as_points",
    "as_array", "as_matrix", "as_data_frame",
    # spatvec
    "geomtype", "is_lines", "is_polygons", "is_points",
    "geom", "crds",
    "expanse", "perim", "nseg",
    "fill_holes", "vect_as_df", "geom_as_wkt",
    # dimensions
    "nrow", "ncol", "nlyr", "ncell", "res", "origin",
    # helpers
    "spat_options", "deepcopy", "tighten",
    # extent
    "ext_align",
    # raster geometry
    "is_rotated", "is_flipped", "flip", "rotate", "shift", "rescale",
    "trans", "trim", "rev_raster",
    # raster values
    "clamp", "clamp_ts", "classify", "subst", "cover", "diff_raster",
    "disagg", "segregate", "selectRange", "sort_raster",
    "range_fill", "weighted_mean",
    # raster analysis
    "boundaries", "patches", "cellSize", "surfArea", "terrain", "shade", "nidp",
    "sieve", "rectify", "stretch", "scale_linear", "scale_raster",
    "quantile_raster", "atan_2",
    # unified generics + raster processing
    "buffer", "project", "intersect",
    "crop", "mask", "resample",
    # vector
    # scoff
    "scoff", "scoff_set",
    "roll", "thresh", "select_highest", "divide", "approximate", "extract_range",
    # arith (Arith_generics.R)
    "is_na", "not_na", "is_true", "is_false",
    "is_nan", "is_finite", "is_infinite",
    "any_na", "all_na", "no_na", "count_na",
    "which_max", "which_min", "which_lyr",
    "where_max", "where_min",
    "rast_sum", "rast_mean", "rast_min", "rast_max", "global_",
    "rast_median", "rast_modal", "stdev_rast",
    "compare_rast", "logic_rast_fn",
    "as_int_rast", "as_bool_rast",
    "is_bool_rast", "is_int_rast", "is_num_rast",
    # geom (geom.R)
    "is_valid", "make_valid",
    "union", "erase", "symdif",
    "spin",
    "hull", "delaunay", "voronoi", "elongate",
    "merge_lines", "make_nodes", "remove_dup_nodes",
    "simplify_geom", "thin_nodes", "thin",
    "shared_paths", "snap", "gaps",
    "force_ccw", "width", "clearance",
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
    # tile_apply
    "tile_apply", "get_tile_extents", "make_tiles",
    "__version__",
]
