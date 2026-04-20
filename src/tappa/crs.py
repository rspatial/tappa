"""CRS helpers — parallel to R ``terra::crs()``."""

from __future__ import annotations

from typing import Any, List, Optional

from ._helpers import character_crs, messages, _getSpatDF

__all__ = ["crs", "proj_pipelines"]


def crs(x: Any, value: Optional[str] = None, *, proj4: bool = False) -> Any:
    """
    Get or set the coordinate reference system, like R ``terra::crs()``.

    * ``crs(x)`` — return CRS string (WKT by default; use ``proj4=True`` for PROJ.4).
    * ``crs(x, value)`` — set CRS from a string or another object; returns *x*.
    """
    if value is not None:
        s = character_crs(value, "crs")
        x.set_crs(s)
        return messages(x, "crs")

    kind = "proj4" if proj4 else "wkt"
    return x.get_crs(kind)


def proj_pipelines(
    source: Any,
    target: Any,
    authority: str = "",
    AOI: Optional[List[float]] = None,
    use: str = "NONE",
    grid_availability: str = "USED",
    desired_accuracy: float = -1.0,
    strict_containment: bool = False,
    axis_order_authority_compliant: bool = False,
) -> Any:
    """
    Enumerate PROJ coordinate-transformation pipelines, like R ``terra::proj_pipelines()``.

    Returns a ``pandas.DataFrame`` (or ``None`` if pandas is unavailable) with
    columns *description*, *definition*, *has_inverse*, *accuracy*, *grid_count*.

    Parameters
    ----------
    source, target : str or spatial object
        Source and target CRS (WKT, PROJ string, or any object with ``get_crs``).
    authority : str
        Restrict to pipelines from this authority (e.g. ``"EPSG"``).
    AOI : list of float, optional
        Area of interest ``[xmin, ymin, xmax, ymax]`` in degrees.
    use : str
        ``"NONE"`` (default), ``"BOTH"``, ``"INTERSECTION"`` or ``"SMALLEST"``.
    grid_availability : str
        ``"USED"`` (default), ``"DISCARD"`` or ``"IGNORED"``.
    desired_accuracy : float
        Filter by minimum accuracy in metres (``-1`` = no filter).
    strict_containment : bool
        Require strict area containment.
    axis_order_authority_compliant : bool
        Use authority-compliant axis order.
    """
    from ._terra import SpatVector

    if not isinstance(source, str):
        source = source.get_crs("wkt") if hasattr(source, "get_crs") else str(source)
    if not isinstance(target, str):
        target = target.get_crs("wkt") if hasattr(target, "get_crs") else str(target)

    if not source:
        raise ValueError("source CRS is empty")
    if not target:
        raise ValueError("target CRS is empty")

    aoi = list(AOI) if AOI is not None else []

    v = SpatVector()
    sdf = v.get_proj_pipelines(
        source, target, authority, aoi, use,
        grid_availability, desired_accuracy,
        strict_containment, axis_order_authority_compliant,
    )
    messages(v, "proj_pipelines")

    return _getSpatDF(sdf)
