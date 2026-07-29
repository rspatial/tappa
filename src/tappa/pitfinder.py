"""
pitfinder — locate pits (flow sinks) from a flow-direction raster (R ``pitfinder``).
"""
from __future__ import annotations

from ._terra import SpatRaster
from ._helpers import messages, spatoptions


def pitfinder(
    x: SpatRaster,
    *,
    pits_on_boundary: bool = True,
    filename: str = "",
    overwrite: bool = False,
) -> SpatRaster:
    """
    Find pit cells from a flow-direction raster.

    Parameters
    ----------
    x : SpatRaster
        Flow directions (e.g. from ``terrain(..., 'flowdir')``).
    pits_on_boundary : bool
        If ``True``, cells on the raster boundary can be pits.

    Returns
    -------
    SpatRaster
        Non-zero values mark pits (see R ``pitfinder``).
    """
    opt = spatoptions(filename, overwrite)
    xc = x.pitfinder2(int(pits_on_boundary), opt)
    return messages(xc, "pitfinder")
