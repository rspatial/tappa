"""
Unified generics — R-style dispatch on the type of the first argument.
"""

from __future__ import annotations

from typing import Any, Union

from ._terra import SpatRaster, SpatVector

__all__ = [
    "buffer",
    "project",
    "intersect",
]


def buffer(x: Union[SpatRaster, SpatVector], width: Any, **kwargs: Any):
    """
    Buffer a :class:`SpatRaster` or :class:`SpatVector` — like R ``buffer()``.
    """
    if isinstance(x, SpatRaster):
        from .distance import _buffer_rast
        return _buffer_rast(x, width, **kwargs)
    if isinstance(x, SpatVector):
        from .geom import _buffer_vect
        return _buffer_vect(x, width, **kwargs)
    raise TypeError(
        f"buffer: expected SpatRaster or SpatVector, got {type(x).__name__}"
    )


def project(
    x: Union[SpatRaster, SpatVector],
    y: Any,
    **kwargs: Any,
):
    """
    Reproject a :class:`SpatRaster` or :class:`SpatVector` — like R ``project()``.
    """
    if isinstance(x, SpatRaster):
        from .generics import _project_raster
        return _project_raster(x, y, **kwargs)
    if isinstance(x, SpatVector):
        from .generics import _project_vector
        return _project_vector(x, y, **kwargs)
    raise TypeError(
        f"project: expected SpatRaster or SpatVector, got {type(x).__name__}"
    )


def intersect(
    x: Union[SpatRaster, SpatVector],
    y: Any,
    **kwargs: Any,
):
    """
    Intersect — like R ``intersect()``.

    For rasters, *y* must be a :class:`SpatRaster`.
    For vectors, *y* may be a :class:`SpatVector` or :class:`SpatExtent`.
    """
    if isinstance(x, SpatRaster):
        if not isinstance(y, SpatRaster):
            raise TypeError("intersect: raster intersection requires a SpatRaster")
        from .generics import _intersect_rast
        return _intersect_rast(x, y)
    if isinstance(x, SpatVector):
        from .geom import _intersect_vect
        return _intersect_vect(x, y, **kwargs)
    raise TypeError(
        f"intersect: expected SpatRaster or SpatVector, got {type(x).__name__}"
    )
