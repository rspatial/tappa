"""
Tessellation — parallel to R ``terra::tessellate()``.

Create a tessellation of polygons that cover an extent with no gaps or
overlaps.  Cells can be hexagons, rectangles, or Goldberg polyhedron faces.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ._helpers import character_crs, messages
from ._terra import SpatExtent, SpatVector

__all__ = ["tessellate"]


def _extent_from(x: Any) -> SpatExtent:
    """Return a SpatExtent from anything with ``.extent()`` or ``vector``."""
    if isinstance(x, SpatExtent):
        return x
    if hasattr(x, "extent") and callable(getattr(x, "extent")):
        e = x.extent()
        if isinstance(e, SpatExtent):
            return e
    if hasattr(x, "vector"):
        v = list(x.vector)
        if len(v) == 4:
            return SpatExtent(float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    raise TypeError("tessellate: cannot extract a SpatExtent from x")


def _crs_from(x: Any) -> str:
    """Return the WKT CRS string of *x* (empty if none)."""
    if hasattr(x, "get_crs"):
        try:
            return x.get_crs("wkt") or ""
        except Exception:
            return ""
    return ""


def _is_lonlat_crs(crs: str, perhaps: bool = True) -> bool:
    """Return True if *crs* (WKT or PROJ string) describes a geographic CRS."""
    if not crs:
        return False
    s = crs.lower()
    if "+proj=longlat" in s or "+proj=latlong" in s:
        return True
    if "geogcs" in s or "geogcrs" in s or "geodcrs" in s:
        if "projcs" in s or "projcrs" in s:
            return False
        return True
    return False


def _looks_like_lonlat_extent(e: SpatExtent) -> bool:
    """Heuristic: extent coordinates are within plausible lon/lat ranges."""
    v = list(e.vector)
    xmin, xmax, ymin, ymax = float(v[0]), float(v[1]), float(v[2]), float(v[3])
    return (-180.01 <= xmin and xmax <= 360.01
            and -90.01 <= ymin and ymax <= 90.01)


def tessellate(
    x: Any = None,
    size: Optional[float] = None,
    n: Optional[int] = None,
    type: str = "hexagon",
    flat_top: bool = False,
    align: str = "fit",
    geo: Optional[bool] = None,
) -> SpatVector:
    """
    Create a tessellation of polygons covering an extent.

    Parallel to R ``terra::tessellate()``.

    Parameters
    ----------
    x : SpatExtent, SpatRaster, SpatVector, or None
        Object from which a :class:`SpatExtent` can be extracted.  If
        ``None`` a global lon/lat extent (``-180, 180, -90, 90``) is used.
    size : float, optional
        Across-flats distance of a hexagon, or center-to-center distance
        between rectangles in the dominant direction.  In CRS units
        (metres for lon/lat).  Approximate for polyhedrons.
    n : int, optional
        Polyhedron subdivision frequency.  Output has ``10*n**2 + 2``
        cells (12 pentagons + ``10*(n**2 - 1)`` hexagons).  When
        provided, *size* is ignored.
    type : str, default ``"hexagon"``
        One of ``"hexagons"``, ``"rectangles"``, or ``"polyhedron"``
        (singular and plural forms accepted; case-insensitive prefix
        matching, like R's ``match.arg``).
    flat_top : bool, default False
        If True, hexagons have two horizontal (flat) edges; if False
        (default) they have two vertical edges with a vertex pointing
        up and down (pointy-top).
    align : str, default ``"fit"``
        Rectangle alignment, one of ``"fit"``, ``"equal"``, ``"cube"``.
    geo : bool, optional
        If True, treat the extent as longitude/latitude coordinates.
        If None (default) the CRS is guessed from the coordinates.
        If False the CRS is set to ``"local"`` if *x* has no CRS.

    Returns
    -------
    SpatVector of polygons.
    """
    # --- option parsing (R match.arg semantics: prefix, case-insensitive) ----
    type_choices = ("hexagons", "rectangles", "polyhedron")
    t = (type or "").strip().lower()
    # Permissive matching: accept either prefix direction (so "hexagon",
    # "hexagons", "polyhedron", and "polyhedrons" all work, mirroring how
    # the R test suite calls tessellate(type="polyhedrons")).
    matches = [c for c in type_choices if c.startswith(t) or t.startswith(c)]
    if len(matches) != 1:
        raise ValueError(
            f"tessellate: type must be one of {type_choices}, got {type!r}"
        )
    type_ = matches[0]

    # --- extract extent and CRS ---------------------------------------------
    globe = x is None
    if globe:
        e = SpatExtent(-180.0, 180.0, -90.0, 90.0)
        crs = "lonlat"
    elif isinstance(x, SpatExtent):
        e = x
        if geo is True:
            crs = "lonlat"
        elif geo is None:
            crs = "lonlat" if _looks_like_lonlat_extent(e) else "local"
        else:
            crs = "local"
    else:
        crs = _crs_from(x) or ""
        try:
            e = _extent_from(x)
        except Exception as exc:
            raise RuntimeError(
                f"tessellate: cannot extract a SpatExtent from x ({exc})"
            )

    # --- size / n validation -------------------------------------------------
    use_n = (type_ == "polyhedron") and (n is not None)
    if use_n:
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            raise ValueError("tessellate: n must be an integer >= 1")
        if n_int < 1:
            raise ValueError("tessellate: n must be >= 1")
    else:
        if (size is None
                or not isinstance(size, (int, float))
                or isinstance(size, bool)
                or not math.isfinite(float(size))
                or float(size) <= 0):
            raise ValueError("tessellate: size must be a single positive number")
        size_f = float(size)

    crs_str = character_crs(crs, "tessellate")
    if not crs_str and geo is True:
        crs_str = "+proj=longlat"

    is_lonlat = _is_lonlat_crs(crs_str, perhaps=True)

    v = SpatVector()

    if is_lonlat:
        if type_ == "polyhedron":
            if not use_n:
                # derive n from desired hex area (matches R wrapper)
                R = 6378137.0
                C_target = 8.0 * math.pi * R * R / (math.sqrt(3.0) * size_f * size_f)
                n_int = int(round(math.sqrt(max(1.0, (C_target - 2.0) / 10.0))))
            n_int = max(1, n_int)
            v = v.polyhedron(e, int(n_int), bool(globe))
        elif type_ == "rectangles":
            align_choices = ("fit", "equal", "cube")
            a = (align or "").strip().lower()
            am = [c for c in align_choices if c.startswith(a)]
            if len(am) != 1:
                raise ValueError(
                    f"tessellate: align must be one of {align_choices}, got {align!r}"
                )
            align_int = align_choices.index(am[0])
            v = v.rectangles_lonlat(e, size_f, int(align_int))
        else:  # hexagons
            v = v.hexagons_lonlat(e, size_f, bool(flat_top))
    else:
        if type_ == "polyhedron":
            raise RuntimeError(
                "tessellate: polyhedron is only available for lon/lat data"
            )
        if type_ == "rectangles":
            # Planar rectangles: degenerate to as.polygons(rast(e, res=sqrt(size)))
            from .rast import rast as _rast
            from .coerce import as_polygons as _as_polygons
            r = _rast(e, resolution=math.sqrt(size_f),
                      crs=(crs_str if crs_str else None))
            return _as_polygons(r)
        # planar hexagons
        v = v.hexagons(e, size_f, crs_str, bool(flat_top),
                       float("nan"), float("nan"))

    return messages(v, "tessellate")
