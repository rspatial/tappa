"""
methods.py — register high-level Python functions as methods on the C++ types.

After :func:`registerMethods` is called (done once at import time by
``terra/__init__.py``), the Python-level functions from ``generics.py``,
``values.py``, ``cells.py``, etc. are accessible directly on objects:

    r.crop(e)           # instead of terra.crop(r, e)
    r.mask(mask_r)      # instead of terra.mask(r, mask_r)
    r.values()          # instead of terra.values(r)
    r.aggregate(2)      # instead of terra.aggregate(r, 2)
    v.buffer(1000)      # instead of terra.buffer(v, 1000)
    e.intersect(e2)     # SpatExtent.intersect is already C++; this is the Python alias

Only the **Python-level** wrappers are registered here.  The raw C++ methods
(e.g. ``r.classify(...)``) remain accessible but may require different argument
conventions — the registered wrappers follow the R-like Python API.
"""
from __future__ import annotations

from ._terra import SpatRaster, SpatVector, SpatExtent


def registerMethods() -> None:
    """
    Attach high-level Python functions as methods on SpatRaster, SpatVector,
    and SpatExtent.

    Called once at import time by ``terra/__init__.py``.
    """

    # ── delayed imports to avoid circular deps ────────────────────────────────
    # We import inside the function so this module can be imported at any time
    # without triggering circular import issues.

    # ── SpatRaster methods ────────────────────────────────────────────────────

    from .dispatch import buffer, project, intersect
    from .generics import (
        crop, mask, resample,
        classify, subst, clamp, clampTS, cover, diffRaster,
        boundaries, patches, cellSize, surfArea, terrain,
        sieve, stretch, scaleLinear, scaleRaster,
        trim, flip, rotate, shift, rescale, revRaster, sortRaster,
        disagg, segregate, selectRange, rangeFill, weightedMean,
        quantileRaster, atan_2,
    )
    from .values import (
        values, set_values, focalValues,
        has_values, in_memory, sources,
        has_min_max, minMax, setMinMax, compareGeom,
    )
    from .cells import (
        cells, cellFromXY, cellFromRowCol,
        xyFromCell, rowColFromCell,
        rowFromY, colFromX,
    )
    from .aggregate import aggregate
    from .focal import focal, focal3D, focalMat
    from .zonal import zonal
    from .flowAccumulation import flowAccumulation
    from .pitfinder import pitfinder
    from .extract import extract
    from .app import app, lapp, tapp, sapp
    from .math import (
        math as rast_math, log, sqrt,
        abs_ as rast_abs, ceiling, floor, round_,
        cumsum, cumprod, cummax, cummin,
        ifel,
    )
    from .distance import distance, costDist, gridDist
    from .rasterize import rasterize
    from .sample import spatSample
    from .window import has_window, setWindow, removeWindow, extend
    from .init import init
    from .arith import (
        is_na, not_na, any_na, all_na, no_na, count_na,
        is_nan, is_finite, is_infinite,
        whichMax, whichMin, whichLyr,
        whereMax, whereMin,
        rast_sum, rast_mean, rast_min, rast_max,
        rast_median, rast_modal, stdevRast,
        compareRast,
    )
    from .levels import (
        is_factor, asFactor,
        levels, setLevels,
        cats, setCats, dropLevels, catalyze,
        has_colors, coltab, setColtab,
    )
    from .names import (
        varnames, setVarnames, longnames, setLongnames,
    )
    from .time import has_time, timeInfo, getTime, setTime
    from .write import write, update
    from .stats import row_sums, col_sums, row_means, col_means, autocor, layerCor
    from .merge import merge, mosaic
    from .coerce import asPolygons, asLines, asPoints, asArray, asMatrix, asDataFrame
    from .plot import plot, plot_rgb
    from .tileApply import tileApply, getTileExtents, makeTiles

    _rast_methods = {
        # extent (R-style; backed by SpatExtent.vector = xmin, xmax, ymin, ymax)
        "xmin":           lambda self: float(self.extent.vector[0]),
        "xmax":           lambda self: float(self.extent.vector[1]),
        "ymin":           lambda self: float(self.extent.vector[2]),
        "ymax":           lambda self: float(self.extent.vector[3]),
        # geometry / metadata
        "crop":           lambda self, y, **kw: crop(self, y, **kw),
        "mask":           lambda self, m, **kw: mask(self, m, **kw),
        "buffer":         lambda self, width, **kw: buffer(self, width, **kw),
        "resample":       lambda self, y, **kw: resample(self, y, **kw),
        "project":        lambda self, crs, **kw: project(self, crs, **kw),
        "trim":           lambda self, **kw: trim(self, **kw),
        "flip":           lambda self, **kw: flip(self, **kw),
        "rotate":         lambda self, **kw: rotate(self, **kw),
        "shift":          lambda self, **kw: shift(self, **kw),
        "rev":            lambda self, **kw: revRaster(self, **kw),
        "aggregate":      lambda self, fact, fun="mean", **kw: aggregate(self, fact, fun, **kw),
        "disagg":         lambda self, fact, **kw: disagg(self, fact, **kw),
        # values
        "values":         lambda self: values(self),
        "setValues":     lambda self, v: set_values(self, v),
        "focalValues":   lambda self, w=3, **kw: focalValues(self, w, **kw),
        "has_values":     lambda self: has_values(self),
        "in_memory":      lambda self: in_memory(self),
        "minMax":        lambda self: minMax(self),
        # classification / recoding
        "classify":       lambda self, rcl, **kw: classify(self, rcl, **kw),
        "subst":          lambda self, from_v, to_v, **kw: subst(self, from_v, to_v, **kw),
        "clamp":          lambda self, lower, upper, **kw: clamp(self, lower, upper, **kw),
        "ifel":           lambda self, test, false_val, **kw: ifel(self, test, false_val, **kw),
        # raster analysis
        "boundaries":     lambda self, **kw: boundaries(self, **kw),
        "patches":        lambda self, **kw: patches(self, **kw),
        "terrain":        lambda self, v="slope", **kw: terrain(self, v, **kw),
        "flowAccumulation": lambda self, weight=None, **kw: flowAccumulation(
            self, weight=weight, **kw
        ),
        "pitfinder":    lambda self, **kw: pitfinder(self, **kw),
        "focal":          lambda self, w, fun="sum", **kw: focal(self, w, fun, **kw),
        "focal3D":        lambda self, w, fun="sum", **kw: focal3D(self, w, fun, **kw),
        "zonal":          lambda self, z, fun="mean", **kw: zonal(self, z, fun, **kw),
        "expanse":        lambda self, **kw: expanse(self, **kw),
        "app":            lambda self, fun, **kw: app(self, fun, **kw),
        "lapp":           lambda self, fun, **kw: lapp(self, fun, **kw),
        "tapp":           lambda self, index, fun, **kw: tapp(self, index, fun, **kw),
        "sapp":           lambda self, fun, **kw: sapp(self, fun, **kw),
        "extract":        lambda self, y, **kw: extract(self, y, **kw),
        "rasterize":      lambda self, v, **kw: rasterize(v, self, **kw),
        # cells / coordinates
        "cells":          lambda self, y=None, **kw: cells(self, y, **kw),
        "xyFromCell":   lambda self, cell: xyFromCell(self, cell),
        "cellFromXY":   lambda self, xy: cellFromXY(self, xy),
        # math
        "log":            lambda self, base=None: log(self) if base is None else log(self, base),
        "sqrt":           lambda self: sqrt(self),
        "abs":            lambda self: rast_abs(self),
        "ceiling":        lambda self: ceiling(self),
        "floor":          lambda self: floor(self),
        "round":          lambda self, digits=0: round_(self, digits),
        "cumsum":         lambda self: cumsum(self),
        "stretch":        lambda self, **kw: stretch(self, **kw),
        # NA
        "is_na":          lambda self: is_na(self),
        "not_na":         lambda self: not_na(self),
        "count_na":       lambda self: count_na(self),
        # summaries (global)
        "sum":            lambda self, na_rm=False: rast_sum(self, na_rm=na_rm),
        "mean":           lambda self, na_rm=False: rast_mean(self, na_rm=na_rm),
        "min":            lambda self, na_rm=False: rast_min(self, na_rm=na_rm),
        "max":            lambda self, na_rm=False: rast_max(self, na_rm=na_rm),
        "sd":             lambda self, na_rm=False: stdevRast(self, na_rm=na_rm),
        # levels / categories
        "levels":         lambda self: levels(self),
        "setLevels":     lambda self, v: setLevels(self, v),
        "cats":           lambda self: cats(self),
        "is_factor":      lambda self: is_factor(self),
        "asFactor":      lambda self: asFactor(self),
        "catalyze":       lambda self: catalyze(self),
        # time
        "has_time":       lambda self: has_time(self),
        "getTime":       lambda self: getTime(self),
        "setTime":       lambda self, v, **kw: setTime(self, v, **kw),
        # write / update
        "write":          lambda self, filename, **kw: write(self, filename, **kw),
        "update":         lambda self, **kw: update(self, **kw),
        # stats
        "autocor":        lambda self, **kw: autocor(self, **kw),
        "layerCor":      lambda self, fun="cor", **kw: layerCor(self, fun, **kw),
        "rowSums":       lambda self, **kw: row_sums(self, **kw),
        "colSums":       lambda self, **kw: col_sums(self, **kw),
        # merge
        "merge":          lambda self, *others, **kw: merge(self, *others, **kw),
        "mosaic":         lambda self, *others, **kw: mosaic(self, *others, **kw),
        # sampling
        "sample":         lambda self, size, **kw: spatSample(self, size, **kw),
        # window
        "window":         lambda self: has_window(self),
        "setWindow":     lambda self, e: setWindow(self, e),
        "removeWindow":  lambda self: removeWindow(self),
        "extend":         lambda self, y, **kw: extend(self, y, **kw),
        # coerce
        "asPolygons":    lambda self, **kw: asPolygons(self, **kw),
        "asPoints":      lambda self, **kw: asPoints(self, **kw),
        "asArray":       lambda self, **kw: asArray(self, **kw),
        "asMatrix":      lambda self, **kw: asMatrix(self, **kw),
        "asDataFrame":  lambda self, **kw: asDataFrame(self, **kw),
        # plot
        "plot":           lambda self, **kw: plot(self, **kw),
        "plotRGB":       lambda self, **kw: plot_rgb(self, **kw),
        # tiles
        "tileApply":      lambda self, fun, cores=1, **kw: tileApply(self, fun, cores, **kw),
        "getTileExtents": lambda self, y=None, **kw: getTileExtents(self, y, **kw),
        "makeTiles":      lambda self, y, **kw: makeTiles(self, y, **kw),
    }

    # Methods to always register (Python wrappers that supersede the C++ raw method).
    # NOTE: after the camelCase rename some Python wrappers gained the same name
    # as the C++ pybind binding (setValues, asPolygons, asPoints, asLines,
    # asFactor, makeTiles, …). Forcing the override here used to cause infinite
    # recursion (the wrapper called e.g. ``y.setValues(...)``, which dispatched
    # back through the lambda). Names that exist as C++ methods are now left
    # alone — use ``pt.setValues(r, v)``, ``pt.asPolygons(r)``, etc. for the
    # Python wrapper conveniences (broadcasting, kwarg defaults).
    _rast_force = {
        "crop", "mask", "buffer", "project", "names", "classify", "values",
        "aggregate", "focal", "zonal", "extract", "app", "lapp", "tapp", "sapp",
        "merge", "mosaic", "write", "plot",
        "levels", "cats", "is_factor",
        "is_na", "not_na", "sum", "mean", "min", "max",
    }
    for name, fn in _rast_methods.items():
        if name in _rast_force or not hasattr(SpatRaster, name):
            setattr(SpatRaster, name, fn)

    # ── SpatVector methods ────────────────────────────────────────────────────

    from .generics import (
        shift, rotate, rescale, trans, flip, disagg,
    )
    from .geom import (
        is_valid, make_valid,
        union, erase, symdif,
        spin,
        hull, delaunay, voronoi,
        simplify_geom, thin_nodes, thin, gaps,
        is_empty,
    )
    from .distance import distance
    from .aggregate import aggregate
    from .spatvec import (
        geomtype, is_lines, is_polygons, is_points,
        geom, crds,
        expanse, perim, nseg,
        fillHoles, vectAsDF, geomAsWkt,
    )
    from .relate import is_related, relate, relate_self
    from .rasterize import rasterize as rasterize_fn
    from .extract import extract as extract_fn
    from .merge import merge
    from .write import write
    _vect_methods = {
        "project":        lambda self, crs, **kw: project(self, crs, **kw),
        "crop":           lambda self, y, **kw: crop(self, y, **kw),
        "mask":           lambda self, m, **kw: mask(self, m, **kw),
        "buffer":         lambda self, width, **kw: buffer(self, width, **kw),
        "hull":           lambda self, **kw: hull(self, **kw),
        "voronoi":        lambda self, **kw: voronoi(self, **kw),
        "delaunay":       lambda self, **kw: delaunay(self, **kw),
        "simplify":       lambda self, tol, **kw: simplify_geom(self, tol, **kw),
        "thinNodes":     lambda self, threshold=1e-6, **kw: thin_nodes(self, threshold, **kw),
        "thin":           lambda self, d, **kw: thin(self, d, **kw),
        "is_valid":       lambda self: is_valid(self),
        "makeValid":     lambda self: make_valid(self),
        "is_empty":       lambda self: is_empty(self),
        "union":          lambda self, y=None, **kw: union(self, y, **kw),
        "intersect":      lambda self, y, **kw: intersect(self, y, **kw),
        "erase":          lambda self, y, **kw: erase(self, y, **kw),
        "disagg":         lambda self, **kw: disagg(self, **kw),
        "flip":           lambda self, **kw: flip(self, **kw),
        "rotate":         lambda self, **kw: rotate(self, **kw),
        "shift":          lambda self, **kw: shift(self, **kw),
        "rescale":        lambda self, **kw: rescale(self, **kw),
        "aggregate":      lambda self, *args, **kw: aggregate(self, *args, **kw),
        "spin":           lambda self, angle, **kw: spin(self, angle, **kw),
        "gaps":           lambda self, **kw: gaps(self, **kw),
        "expanse":        lambda self, **kw: expanse(self, **kw),
        "perim":          lambda self: perim(self),
        "geomtype":       lambda self: geomtype(self),
        "crds":           lambda self, **kw: crds(self, **kw),
        "geom":           lambda self, **kw: geom(self, **kw),
        "as_df":          lambda self: vectAsDF(self),
        "as_wkt":         lambda self: geomAsWkt(self),
        "relate":         lambda self, y, relation, **kw: relate(self, y, relation, **kw),
        "is_related":     lambda self, y, relation, **kw: is_related(self, y, relation, **kw),
        "distance":       lambda self, y=None, **kw: distance(self, y, **kw),
        "rasterize":      lambda self, r, **kw: rasterize_fn(self, r, **kw),
        "extract":        lambda self, r, **kw: extract_fn(r, self, **kw),
        "merge":          lambda self, y=None, *more, **kw: merge(self, y, *more, **kw),
        "write":          lambda self, filename, **kw: write(self, filename, **kw),
    }

    _vect_force = {
        "crop", "mask", "buffer", "simplify", "project",
        "union", "intersect", "erase", "merge", "write",
        "rasterize", "extract", "distance",
        "flip", "rotate", "shift", "rescale", "disagg", "aggregate",
    }
    for name, fn in _vect_methods.items():
        if name in _vect_force or not hasattr(SpatVector, name):
            setattr(SpatVector, name, fn)

    # ── SpatExtent methods ────────────────────────────────────────────────────

    from .extent import ext

    _ext_methods = {
        "crop":    lambda self, y, **kw: crop(y, self, **kw),
    }

    for name, fn in _ext_methods.items():
        if not hasattr(SpatExtent, name):
            setattr(SpatExtent, name, fn)

    # ── names property (getter/setter) ────────────────────────────────────────
    # Keep C++ names accessible as ``r.names`` (list) and ``r.names = [...]``
    # without shadowing internal C++ calls via a no-arg method.

    from .names import _cpp_layer_names, _set_names_inplace, _cpp_set_vect_names

    class _SpatRasterNamesProperty:
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return _cpp_layer_names(obj)

        def __set__(self, obj, value):
            _set_names_inplace(obj, [str(v) for v in value])

    class _SpatVectorNamesProperty:
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return _cpp_layer_names(obj)

        def __set__(self, obj, value):
            _cpp_set_vect_names(obj, [str(v) for v in value])

    SpatRaster.names = _SpatRasterNamesProperty()  # type: ignore[assignment]
    SpatVector.names = _SpatVectorNamesProperty()  # type: ignore[assignment]


register_methods = registerMethods
