"""
write.py — write raster and vector data to files.
"""
from __future__ import annotations
import os
from typing import List, Optional, Union

from ._terra import SpatRaster, SpatVector, SpatOptions
from ._helpers import messages, spatoptions

_cpp_vect_write = SpatVector.write  # captured before monkey-patching


def _opt() -> SpatOptions:
    return SpatOptions()


# ---------------------------------------------------------------------------
# SpatRaster write
# ---------------------------------------------------------------------------

def write_raster(
    x: SpatRaster,
    filename: str,
    overwrite: bool = False,
    filetype: Optional[str] = None,
    datatype: str = "FLT4S",
    gdal: Optional[List[str]] = None,
    **kwargs,
) -> SpatRaster:
    """
    Write a SpatRaster to a file.

    Parameters
    ----------
    x : SpatRaster
    filename : str
        Output path.  The format is inferred from the extension
        (e.g. ``".tif"`` → GeoTIFF, ``".nc"`` → NetCDF).
    overwrite : bool
        If False (default), raise an error if *filename* already exists.
    filetype : str, optional
        GDAL driver name (e.g. ``"GTiff"``).  Auto-detected if None.
    datatype : str
        Output data type: ``"FLT4S"`` (float32, default), ``"FLT8S"``
        (float64), ``"INT2S"`` (int16), ``"INT4S"`` (int32), ``"INT1U"``
        (uint8), etc.
    gdal : list of str, optional
        GDAL creation options (e.g. ``["COMPRESS=LZW"]``).
    **kwargs
        Additional options forwarded to SpatOptions.

    Returns
    -------
    SpatRaster  (re-opened from the written file)
    """
    filename = os.path.expanduser(filename.strip())
    if not filename:
        raise ValueError("provide a filename")
    opt = spatoptions(filename, overwrite)
    opt.datatype = datatype
    if filetype:
        opt.filetype = filetype
    if gdal:
        opt.gdal_options = gdal
    xc = x.writeRaster(opt)
    messages(xc, "writeRaster")
    from .rast import rast
    return rast(filename)


def update(
    x: SpatRaster,
    *,
    names: bool = False,
    crs: bool = False,
    extent: bool = False,
    cells: Optional[List[Union[int, float]]] = None,
    values: Optional[List[Union[int, float]]] = None,
    layer: Union[int, List[int]] = 0,
) -> SpatRaster:
    """
    Update metadata or cell values of a file-backed SpatRaster on disk.

    This modifies the file directly without reading the entire raster into
    memory.  For in-memory rasters a warning is issued and no action is taken.

    Parameters
    ----------
    x : SpatRaster
        Must have a file source (not in-memory).
    names : bool
        Update band names in the file to match *x*.
    crs : bool
        Update the coordinate reference system in the file to match *x*.
    extent : bool
        Update the geotransform / extent in the file to match *x*.
    cells : list of int, optional
        Cell numbers to update (1-indexed).  Must be used together with
        *values*.
    values : list of float, optional
        New values for the specified *cells*.  A single value is recycled,
        one value per cell is applied to all target layers, or
        ``len(cells) * len(layer)`` values can be given in layer-major order.
    layer : int or list of int
        Layer(s) to update when writing cell values.  ``0`` (default) means
        all layers.  Positive integers select specific layers (1-indexed).

    Returns
    -------
    SpatRaster  (*x*, returned invisibly)
    """
    import numpy as np

    opt = _opt()

    if cells is not None or values is not None:
        if cells is None or values is None:
            raise ValueError("provide both 'cells' and 'values'")

        cells_arr = np.asarray(cells, dtype=float)
        vals_arr = np.asarray(values, dtype=float).ravel()

        if isinstance(layer, (list, tuple, np.ndarray)):
            layers_list = [int(l) - 1 for l in layer]
        else:
            layer = int(layer)
            if layer > 0:
                layers_list = [layer - 1]
            else:
                layers_list = []

        cells_cpp = (cells_arr - 1).tolist()
        vals_cpp = vals_arr.tolist()
        layers_cpp = [int(l) for l in layers_list]

        x.update_values(cells_cpp, vals_cpp, layers_cpp, opt)
        messages(x, "update")

    if names or crs or extent:
        x.update_meta(names, crs, extent, opt)
        messages(x, "update")

    return x


def write_start(
    x: SpatRaster,
    filename: str,
    overwrite: bool = False,
    n: int = 4,
    sources: Optional[List[str]] = None,
) -> dict:
    """
    Open a file for block-wise writing.

    Parameters
    ----------
    x : SpatRaster
    filename : str
    overwrite : bool
    n : int
        Number of block copies to buffer in memory.
    sources : list of str, optional

    Returns
    -------
    dict with keys ``"n"``, ``"row"``, ``"nrows"``.
    """
    filename = os.path.expanduser(filename.strip())
    opt = spatoptions(filename, overwrite)
    opt.ncopies = n
    if sources is None:
        sources = []
    ok = x.writeStart(opt, list(set(sources)))
    messages(x, "writeStart")
    b = x.getBlockSizeWrite()
    return b


def write_values(
    x: SpatRaster,
    v: List[float],
    start: int,
    nrows: int,
) -> bool:
    """
    Write a block of values to a file opened with write_start().

    Parameters
    ----------
    x : SpatRaster
    v : list of float
    start : int
        Starting row (0-based; matches C++ / ``write_start`` / ``blocks``).
    nrows : int
        Number of rows in this block.

    Returns
    -------
    bool
    """
    ok = x.writeValues(list(v), start, nrows)
    messages(x, "writeValues")
    return bool(ok)


def write_stop(x: SpatRaster) -> SpatRaster:
    """
    Finalise a block-wise write and close the file.

    Parameters
    ----------
    x : SpatRaster

    Returns
    -------
    SpatRaster  (re-opened from the written file)
    """
    x.writeStop()
    messages(x, "writeStop")
    src = list(x.filenames())
    if src and src[0]:
        from .rast import rast
        return rast(src[0])
    return x


def blocks(x: SpatRaster, n: int = 4) -> dict:
    """
    Return the block structure for writing *x*.

    Parameters
    ----------
    x : SpatRaster
    n : int
        Number of copies to buffer.

    Returns
    -------
    dict with keys ``"n"``, ``"row"`` (0-based), ``"nrows"``.
    """
    opt = spatoptions()
    opt.ncopies = n
    b = x.getBlockSizeR(opt)
    return b


# ---------------------------------------------------------------------------
# SpatVector write
# ---------------------------------------------------------------------------

_EXT_TO_FILETYPE = {
    "shp": "ESRI Shapefile",
    "shz": "ESRI Shapefile",
    "gpkg": "GPKG",
    "gdb": "OpenFileGDB",
    "gml": "GML",
    "json": "GeoJSON",
    "geojson": "GeoJSON",
    "cdf": "netCDF",
    "svg": "SVG",
    "kml": "KML",
    "vct": "Idrisi",
    "tab": "MapInfo File",
    "gpx": "GPX",
}


def _guess_filetype(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    ft = _EXT_TO_FILETYPE.get(ext)
    if ft is None:
        raise ValueError(f"cannot guess filetype from filename {filename!r}")
    return ft


def write_vector(
    x: SpatVector,
    filename: str,
    filetype: Optional[str] = None,
    layer: Optional[str] = None,
    insert: bool = False,
    overwrite: bool = False,
    options: str = "ENCODING=UTF-8",
) -> bool:
    """
    Write a SpatVector to a file.

    Parameters
    ----------
    x : SpatVector
    filename : str
        Output path.
    filetype : str, optional
        OGR driver name.  Auto-detected from extension if None.
    layer : str, optional
        Layer name inside the file.  Defaults to the base filename.
    insert : bool
        Append to an existing file instead of replacing.
    overwrite : bool
        Overwrite an existing layer.
    options : str
        OGR creation options string.

    Returns
    -------
    bool
    """
    filename = os.path.expanduser(filename.strip())
    if not filename:
        raise ValueError("provide a filename")
    if filetype is None:
        filetype = _guess_filetype(filename)
    if layer is None:
        layer = os.path.splitext(os.path.basename(filename))[0].strip()

    # Truncate field names for Shapefiles (max 10 chars)
    if filetype == "ESRI Shapefile":
        nms = list(x.names)
        truncated = [n[:10] for n in nms]
        if truncated != nms:
            xc = x.deepcopy()
            xc.names = truncated
            x = xc

    ok = _cpp_vect_write(x, filename, layer, filetype, insert, overwrite, options)
    messages(x, "writeVector")
    return bool(ok)
