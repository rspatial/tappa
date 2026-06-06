"""Create SpatVectors — parallel to R ``terra::vect()``."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from ._helpers import character_crs, messages
from ._terra import SpatExtent, SpatVector

__all__ = ["vect"]


def _looks_like_wkt(s: str) -> bool:
    # Match R's first-five-character test (POINT, MULTI, LINES, POLYG, EMPTY)
    t = s.strip().upper()[:5]
    return t in ("POINT", "MULTI", "LINES", "POLYG", "EMPTY") or s.strip().startswith("{")


def _vect_xy_matrix(x: Any, crs: str) -> SpatVector:
    """
    Build points from an (n, 2) matrix — same as R ``vect(matrix)`` (``R/vect.R``):
    ``SpatVector()`` → set CRS → ``setPointsXY(x[,1], x[,2])``.
    Not WKT strings (that path can differ internally from R).
    """
    import numpy as np

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("vect: coordinate matrix must have shape (n, 2) with columns [x, y]")
    v = SpatVector()
    if crs:
        v.set_crs(character_crs(crs, "vect"))
    v.setPointsXY(arr[:, 0].tolist(), arr[:, 1].tolist())
    return messages(v, "vect")


def _vect_geom_matrix(x: Any, geom_type: str, crs: str) -> SpatVector:
    """Build a SpatVector of lines or polygons from a ``[id, part, x, y(, hole)]``
    matrix — analogue of R ``terra::vect(matrix, type=...)``.

    Hole flags default to 0. The first three columns are coerced to int /
    float as appropriate.
    """
    import numpy as np

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] not in (4, 5):
        raise ValueError(
            "vect: lines/polygons matrix must have 4 columns (id, part, x, y) "
            "or 5 columns (id, part, x, y, hole)"
        )
    if geom_type not in ("lines", "polygons"):
        raise ValueError("vect: type must be 'lines' or 'polygons'")

    ids = arr[:, 0].astype(int).tolist()
    parts = arr[:, 1].astype(int).tolist()
    xs = arr[:, 2].tolist()
    ys = arr[:, 3].tolist()
    if arr.shape[1] == 5:
        holes = [bool(h) for h in arr[:, 4]]
    else:
        holes = [False] * arr.shape[0]

    v = SpatVector()
    if crs:
        v.set_crs(character_crs(crs, "vect"))
    v.setGeometry(geom_type, ids, parts, xs, ys, holes)
    return messages(v, "vect")


def _normalize_path(path: str) -> str:
    p = path.strip()
    if p.startswith("http") and (p.endswith(".shp") or p.endswith(".gpkg")):
        return "/vsicurl/" + p
    if p.startswith("s3://"):
        return "/vsis3/" + p[5:]
    if os.path.isfile(p):
        try:
            return os.path.abspath(p)
        except OSError:
            return p
    return p


def vect(
    x: Any = None,
    *,
    type: Optional[str] = None,
    layer: str = "",
    query: str = "",
    crs: str = "",
    **kwargs: Any,
) -> SpatVector:
    """
    Create a :class:`SpatVector`, like R ``terra::vect()``.

    * ``vect()`` — empty vector.
    * ``vect(str)`` — WKT literal or path to a vector file (via GDAL).
    * ``vect(SpatExtent)`` — rectangle as polygon (use **crs**).
    * ``list[str]`` — multiple WKT geometries.
    * A coordinate matrix *(n, 2)* — same as R ``vect(matrix)`` via ``setPointsXY``.
    * A geometry matrix *(n, 4)* with columns ``[id, part, x, y]`` (or
      *(n, 5)* with a trailing ``hole`` flag) plus ``type='lines'`` or
      ``type='polygons'`` — same as R ``vect(matrix, type=...)``.

    Extra GDAL arguments (``layer``, ``query``, …) match the C++ ``read`` call
    where applicable; see R ``terra::vect`` for full options (not all are wired yet).
    """
    del kwargs  # reserved for future parity

    if x is None:
        v = SpatVector()
        return messages(v, "vect")

    if isinstance(x, SpatExtent):
        crs_use = character_crs(crs, "vect") if crs else ""
        v = SpatVector(x, crs_use)
        return messages(v, "vect")

    if isinstance(x, str):
        s = x.strip()
        if _looks_like_wkt(s):
            v = SpatVector([s.replace("\n", "")])
            if crs:
                v.set_crs(character_crs(crs, "vect"))
            return messages(v, "vect")

        path = _normalize_path(s)
        v = SpatVector()
        ext: List[float] = []
        filt = SpatVector()
        opts: List[str] = []
        ok = v.read(path, layer, query, ext, filt, False, "", "", opts)
        if not ok:
            messages(v, "vect")
        if crs:
            v.set_crs(character_crs(crs, "vect"))
        return messages(v, "vect")

    if isinstance(x, (list, tuple)) and all(isinstance(i, str) for i in x):
        v = SpatVector([s.replace("\n", "") for s in x])
        if crs:
            v.set_crs(character_crs(crs, "vect"))
        return messages(v, "vect")

    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore
    if np is not None and isinstance(x, np.ndarray):
        if x.ndim == 2 and x.shape[1] == 2:
            if type is not None and type not in ("points", None):
                raise ValueError(
                    f"vect: type={type!r} requires 4 columns [id, part, x, y]"
                )
            return _vect_xy_matrix(x, crs)
        if x.ndim == 2 and x.shape[1] in (4, 5):
            gt = (type or "points").lower()
            if gt in ("lines", "polygons"):
                return _vect_geom_matrix(x, gt, crs)
        raise TypeError(
            "vect: ndarray must be (n, 2) for points, or (n, 4)/(n, 5) "
            "with type='lines'/'polygons'"
        )

    if (
        isinstance(x, (list, tuple))
        and x
        and isinstance(x[0], (list, tuple))
        and len(x[0]) == 2
    ):
        return _vect_xy_matrix(x, crs)

    raise TypeError(
        "vect: use None, str (path or WKT), list[str] (WKT), SpatExtent, "
        "or a coordinate matrix (n, 2)"
    )
