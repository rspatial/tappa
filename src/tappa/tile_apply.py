"""
tile_apply.py — apply a function to spatial tiles of a SpatRaster, in
parallel, and assemble the per-tile results back into a single raster.

Mirrors ``R/tile_apply.R`` (and the auto-tiling helpers from ``R/tiles.R``)
in the terra R package. The pipeline is memory-safe: every per-tile
result is written straight to disk by the worker (or by the sequential
loop) and the parent process only sees filenames, so peak RAM per
process is one tile's worth of values - independent of how big the input
raster is. Assembly is either via ``vrt()`` (free; non-overlapping
tiles, the default) or via a streaming ``mosaic()`` (when ``overlap_fun``
is given).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from ._helpers import messages, spatoptions
from ._terra import SpatExtent, SpatOptions, SpatRaster, SpatVector

__all__ = ["tile_apply", "get_tile_extents", "make_tiles"]


# methods.py registers high-level Python wrappers under these same names on
# SpatRaster, so capture the raw C++ methods up front (mirrors focal.py).
_cpp_get_tiles_ext = SpatRaster.get_tiles_ext
_cpp_get_tiles_ext_vect = SpatRaster.get_tiles_ext_vect
_cpp_make_tiles = SpatRaster.make_tiles
_cpp_make_tiles_vect = SpatRaster.make_tiles_vect


# ---------------------------------------------------------------------------
# Auto tile size
# ---------------------------------------------------------------------------

def _file_blocksize(x: SpatRaster) -> Optional[np.ndarray]:
    """Return the GDAL block size matrix (rows, cols) per source, or None."""
    try:
        v = x.getFileBlocksize()
    except Exception:
        return None
    if v is None or len(v) == 0:
        return None
    arr = np.asarray(v, dtype=int).reshape(-1, 2, order="F")
    if arr.size == 0:
        return None
    return arr


def _free_ram_bytes() -> Optional[float]:
    """Best-effort free-RAM estimate in bytes; None when unavailable."""
    try:
        import psutil  # type: ignore
        return float(psutil.virtual_memory().available)
    except Exception:
        pass
    # Linux fallback: parse /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return float(parts[1]) * 1024.0  # kB -> B
    except Exception:
        pass
    return None


def _auto_tile_size(
    x: SpatRaster,
    cores: int = 1,
    memfrac: Optional[float] = None,
    tasks_per_worker: int = 4,
    ncopies: int = 4,
) -> List[int]:
    """
    Choose a sensible (rows, cols) tile size for *x*, mirroring
    ``.auto_tile_size`` in ``R/tiles.R``:

    - Align tiles to whole GDAL blocks when *x* reports them; fall back to
      256-cell squares otherwise.
    - Cap per-tile cell count by a memory budget so that *cores* workers
      fit within ``memfrac`` of free RAM.
    - Aim for a few tiles per worker for load balancing.
    - Clamp to the raster's own dimensions.
    """
    nr, nc, nl = x.nrow(), x.ncol(), x.nlyr()
    cores = max(int(cores), 1)
    tasks_per_worker = max(int(tasks_per_worker), 1)

    fb = _file_blocksize(x)
    if fb is None:
        br = bc = 0
    else:
        br = int(fb[:, 0].min())
        bc = int(fb[:, 1].min())
    if br <= 0:
        br = min(256, nr)
    if bc <= 0:
        bc = min(256, nc)
    br = min(br, nr); bc = min(bc, nc)
    one_block = float(br) * bc

    bytes_per_cell = 8 * max(nl, 1) * max(int(ncopies), 1)

    if memfrac is None:
        try:
            memfrac = float(SpatOptions().memfrac)
        except Exception:
            memfrac = 0.5
        if not np.isfinite(memfrac) or memfrac <= 0:
            memfrac = 0.5

    free_b = _free_ram_bytes()
    if free_b is None or not np.isfinite(free_b) or free_b <= 0:
        mem_cells = float("inf")
    else:
        budget = (free_b * memfrac) / cores
        mem_cells = budget / bytes_per_cell

    total_cells = float(nr) * nc
    target_min_tiles = cores * tasks_per_worker
    par_cells = total_cells / target_min_tiles

    cells_per_tile = min(mem_cells, par_cells)
    cells_per_tile = max(cells_per_tile, one_block)
    cells_per_tile = min(cells_per_tile, total_cells)

    side = cells_per_tile ** 0.5
    nbr = max(round(side / br), 1)
    nbc = max(round(side / bc), 1)
    tile_rows = min(int(nbr * br), nr)
    tile_cols = min(int(nbc * bc), nc)
    return [tile_rows, tile_cols]


# ---------------------------------------------------------------------------
# get_tile_extents and make_tiles
# ---------------------------------------------------------------------------

def _template_from_factors(x: SpatRaster, fac: Sequence[int]) -> SpatRaster:
    """Build the coarse template raster used by ``get_tiles_ext`` /
    ``make_tiles`` from a per-tile (rows, cols) size, like R's
    ``aggregate(rast(x), y)``.

    ``SpatRaster::geometry(nlyrs, properties, time, units, tags)`` has no
    Python-side default arguments, so we pass the same defaults R's
    ``rast(<SpatRaster>)`` uses.
    """
    from .aggregate import aggregate
    geom_pntr = x.geometry(x.nlyr(), False, True, False, False)
    geom = geom_pntr if isinstance(geom_pntr, SpatRaster) else x.deepcopy()
    return aggregate(geom, [int(fac[0]), int(fac[1])])


def get_tile_extents(
    x: SpatRaster,
    y: Optional[Union[int, Sequence[int], SpatRaster, SpatVector]] = None,
    *,
    extend: bool = False,
    buffer: int = 0,
    cores: int = 1,
) -> np.ndarray:
    """
    Return tile extents for *x* as an ``(N, 4)`` matrix
    ``[xmin, xmax, ymin, ymax]``.

    Parameters
    ----------
    x : SpatRaster
    y : int, [rows, cols], SpatRaster or SpatVector, optional
        Tile geometry. ``int`` or a length-1/2 sequence is interpreted as
        the per-tile size (rows, cols). When ``y`` is omitted the tile
        size is chosen automatically based on the source's GDAL block
        size and a memory budget for *cores* workers (see
        ``tile_apply``).
    extend : bool
        Pad the boundary tiles to the raster extent.
    buffer : int
        Add this many cells of overlap around each tile (used to avoid
        edge effects in focal-style operations).
    cores : int
        Number of workers the tiles are intended for; used only to pick
        a sensible default tile size when ``y`` is omitted.
    """
    if y is None:
        y = _auto_tile_size(x, cores=cores)

    # The C++ get_tiles_extent[*] take buffer as a std::vector<int>; pass
    # a length-1 list so pybind11's sequence converter accepts it.
    buf = [int(buffer)]
    if isinstance(y, SpatRaster):
        e = _cpp_get_tiles_ext(x, y, bool(extend), buf)
    elif isinstance(y, SpatVector):
        e = _cpp_get_tiles_ext_vect(x, y, bool(extend), buf)
    else:
        if isinstance(y, (int, float, np.integer, np.floating)):
            ys = [int(y), int(y)]
        else:
            ys = [int(v) for v in y]
            if len(ys) == 0:
                raise ValueError("get_tile_extents: 'y' is empty")
            if len(ys) > 2:
                raise ValueError("get_tile_extents: expected one or two numbers")
            if len(ys) == 1:
                ys = [ys[0], ys[0]]
        template = _template_from_factors(x, ys)
        e = _cpp_get_tiles_ext(x, template, bool(extend), buf)

    messages(x, "get_tile_extents")
    arr = np.asarray(e, dtype=float).reshape(-1, 4, order="F")
    return arr


def make_tiles(
    x: SpatRaster,
    y: Union[int, Sequence[int], SpatRaster, SpatVector],
    *,
    filename: str = "tile_.tif",
    extend: bool = False,
    na_rm: bool = False,
    buffer: int = 0,
    overwrite: bool = False,
) -> List[str]:
    """
    Write one file per tile to disk and return the list of filenames,
    like R ``makeTiles``. *filename* must contain a placeholder
    (e.g. ``"tile_.tif"``) into which the tile index is inserted.
    """
    fname = filename.strip()
    if not fname:
        raise ValueError("make_tiles: filename cannot be empty")
    opt = spatoptions(filename="", overwrite=overwrite)
    buf = [int(buffer)]
    if isinstance(y, SpatRaster):
        ff = _cpp_make_tiles(x, y, bool(extend), buf, bool(na_rm), fname, opt)
    elif isinstance(y, SpatVector):
        ff = _cpp_make_tiles_vect(x, y, bool(extend), buf, bool(na_rm), fname, opt)
    else:
        if isinstance(y, (int, float, np.integer, np.floating)):
            ys = [int(y), int(y)]
        else:
            ys = [int(v) for v in y]
            if len(ys) == 0:
                raise ValueError("make_tiles: 'y' is empty")
            if len(ys) > 2:
                raise ValueError("make_tiles: expected one or two numbers")
            if len(ys) == 1:
                ys = [ys[0], ys[0]]
        template = _template_from_factors(x, ys)
        ff = _cpp_make_tiles(x, template, bool(extend), buf, bool(na_rm), fname, opt)
    messages(x, "make_tiles")
    return [str(f) for f in ff]


# ---------------------------------------------------------------------------
# Tile extent pairing (outer / inner)
# ---------------------------------------------------------------------------

def _pair(o: Sequence[float], i: Sequence[float]) -> dict:
    return {"outer": [float(v) for v in o], "inner": [float(v) for v in i]}


def _identity_pairs(m: np.ndarray) -> List[dict]:
    return [_pair(m[j], m[j]) for j in range(m.shape[0])]


def _is_extent_4tuple(t: Any) -> bool:
    """True if *t* is a length-4 numeric sequence (but not a SpatExtent)."""
    if isinstance(t, SpatExtent):
        return False
    try:
        if hasattr(t, "__len__") and len(t) == 4:
            return all(isinstance(v, (int, float, np.integer, np.floating)) for v in t)
    except TypeError:
        pass
    return False


def _tile_apply_extents(
    x: SpatRaster,
    tiles: Any,
    cores: int = 1,
    buffer: int = 0,
) -> List[dict]:
    """Return ``[{'outer': [...], 'inner': [...]}, ...]`` for each tile.

    Mirrors ``.tile_apply_extents`` in ``R/tile_apply.R``: for the auto
    path, expand each inner tile by *buffer* cells (clamped to *x*'s
    extent); for explicit tiles trust the caller (outer == inner)."""
    if tiles is None:
        inner = get_tile_extents(x, cores=cores)
        if buffer > 0:
            outer = get_tile_extents(x, cores=cores, buffer=int(buffer))
            ev = list(x.extent.vector)  # xmin, xmax, ymin, ymax
            out: List[dict] = []
            for j in range(inner.shape[0]):
                o = [float(v) for v in outer[j]]
                o[0] = max(o[0], ev[0])
                o[1] = min(o[1], ev[1])
                o[2] = max(o[2], ev[2])
                o[3] = min(o[3], ev[3])
                out.append(_pair(o, inner[j]))
            return out
        return _identity_pairs(inner)

    if isinstance(tiles, SpatExtent):
        v = list(tiles.vector)
        return [_pair(v, v)]

    if isinstance(tiles, np.ndarray):
        if tiles.ndim != 2 or tiles.shape[1] != 4:
            raise ValueError(
                "tile_apply: matrix 'tiles' must have 4 columns: "
                "xmin, xmax, ymin, ymax"
            )
        return _identity_pairs(tiles)

    if isinstance(tiles, list):
        if len(tiles) == 0:
            raise ValueError("tile_apply: 'tiles' is empty")
        # A list of (Extent or length-4 numeric)?
        first = tiles[0]
        if isinstance(first, SpatExtent) or _is_extent_4tuple(first):
            out = []
            for t in tiles:
                if isinstance(t, SpatExtent):
                    v = list(t.vector); out.append(_pair(v, v))
                elif _is_extent_4tuple(t):
                    v = [float(z) for z in t]; out.append(_pair(v, v))
                else:
                    raise ValueError(
                        "tile_apply: list elements of 'tiles' must be a "
                        "SpatExtent or a length-4 numeric "
                        "(xmin, xmax, ymin, ymax)"
                    )
            return out
        # Otherwise treat as a (rows, cols) sequence
        if len(tiles) <= 2 and all(
            isinstance(v, (int, float, np.integer, np.floating)) for v in tiles
        ):
            m = get_tile_extents(x, list(tiles))
            return _identity_pairs(m)
        raise ValueError(
            "tile_apply: 'tiles' list must contain SpatExtents or be a "
            "length-1/2 (rows, cols)"
        )

    if isinstance(tiles, (SpatRaster, SpatVector)):
        m = get_tile_extents(x, tiles)
        return _identity_pairs(m)

    if isinstance(tiles, (int, float, np.integer, np.floating)):
        m = get_tile_extents(x, [int(tiles), int(tiles)])
        return _identity_pairs(m)

    raise ValueError(
        "tile_apply: 'tiles' must be None (auto), a SpatExtent, a list of "
        "SpatExtents, a 4-column numpy matrix of extents, a SpatRaster or "
        "SpatVector defining tile geometry, or one or two numbers (rows, cols)"
    )


# ---------------------------------------------------------------------------
# Worker (must be importable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _serialize_fun(fun: Callable) -> bytes:
    """Pickle *fun* with cloudpickle when available (handles lambdas
    and closures), else stdlib pickle."""
    try:
        import cloudpickle  # type: ignore
        return cloudpickle.dumps(fun)
    except Exception:
        import pickle
        return pickle.dumps(fun)


def _deserialize_fun(blob: bytes) -> Callable:
    try:
        import cloudpickle  # type: ignore
        return cloudpickle.loads(blob)
    except Exception:
        import pickle
        return pickle.loads(blob)


def _tile_worker(args: tuple) -> str:
    """Run on a child process. Returns the per-tile output filename."""
    src, outer, inner, out_file, fun_blob, kw_blob, datatype = args
    # delayed imports so the child only loads what it needs
    from tappa._terra import SpatExtent, SpatRaster as _SR
    from tappa.generics import crop as _crop
    from tappa.rast import rast
    from tappa.window import set_window
    from tappa.write import write_raster

    fun = _deserialize_fun(fun_blob)
    kwargs = _deserialize_fun(kw_blob) if kw_blob else {}

    x = rast(src)
    e = SpatExtent(*outer)
    x = set_window(x, e)
    r = fun(x, **kwargs)
    if not isinstance(r, _SR):
        raise RuntimeError("tile_apply: 'fun' must return a SpatRaster for every tile")
    if outer != inner:
        ie = SpatExtent(*inner)
        r = _crop(r, ie, snap="near")
    if datatype:
        write_raster(r, out_file, overwrite=True, datatype=datatype)
    else:
        write_raster(r, out_file, overwrite=True)
    return out_file


# ---------------------------------------------------------------------------
# Helper: ensure file-backed source for parallel workers
# ---------------------------------------------------------------------------

def _ensure_file_backed(x: SpatRaster) -> tuple:
    """Return ``(src_path, created)`` where *src_path* points to a file
    backing *x*. If *x* is in memory we write it to a temporary GeoTIFF
    and return ``created=True`` so we can clean it up later."""
    # filenames() and inMemory both return one entry per source; reuse the
    # first source if it is a real file on disk.
    src = list(x.filenames())
    in_mem = list(x.inMemory)
    if src and src[0] and (not in_mem or not in_mem[0]):
        return src[0], False
    fd, tmp = tempfile.mkstemp(suffix=".tif", prefix="tile_apply_in_")
    os.close(fd)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    from .write import write_raster
    write_raster(x, tmp, overwrite=True)
    return tmp, True


# ---------------------------------------------------------------------------
# tile_apply
# ---------------------------------------------------------------------------

def tile_apply(
    x: SpatRaster,
    fun: Callable[..., SpatRaster],
    cores: int = 1,
    *,
    tiles: Any = None,
    buffer: int = 0,
    filename: str = "",
    overwrite: bool = False,
    datatype: str = "FLT4S",
    overlap_fun: Optional[str] = None,
    **kwargs: Any,
) -> SpatRaster:
    """
    Apply *fun* to each tile of *x* and assemble the per-tile results.

    Mirrors ``R::terra::tile_apply``. The function is run in a separate
    process per tile when ``cores > 1`` (via
    :class:`concurrent.futures.ProcessPoolExecutor`). Workers always
    write their per-tile result to disk, so the parent process never
    holds more than one tile's worth of pixel data in RAM.

    Parameters
    ----------
    x : SpatRaster
        Input raster. Must not have a window set.
    fun : callable
        ``fun(tile, **kwargs) -> SpatRaster``. *tile* is a windowed
        view onto *x*. The returned raster's extent must match the tile
        (or contain it - it is cropped back to the inner extent for the
        ``buffer`` path).
    cores : int
        Number of worker processes. ``1`` runs sequentially in the
        current process.
    tiles : None, SpatExtent, list of SpatExtents, (N,4) numpy array, \
            SpatRaster, SpatVector, or one or two ints
        How to partition *x*. ``None`` (default) chooses tiles
        automatically based on the source's GDAL block size and a
        memory budget for *cores* workers.
    buffer : int
        Number of cells of overlap to read around each auto-tile (used
        to avoid edge effects in focal-style operations); the per-tile
        output is cropped back to the unbuffered extent before being
        written. Only honoured when ``tiles=None``.
    filename : str
        Output filename. If empty, a VRT-backed SpatRaster pointing at
        the per-tile temporaries is returned (valid for the rest of the
        session).
    overwrite : bool
        Overwrite *filename* if it already exists.
    datatype : str
        Datatype for both the per-tile temporaries and the assembled
        output. Defaults to ``"FLT4S"`` (single-precision float). Use
        ``"FLT8S"`` if you need bit-exact agreement with calling *fun*
        on the whole raster.
    overlap_fun : str, optional
        If given, the per-tile outputs are mosaicked with this function
        instead of stitched with a VRT. Use this when the supplied
        ``tiles`` overlap.
    **kwargs
        Extra keyword arguments forwarded to *fun*.

    Returns
    -------
    SpatRaster
    """
    if not isinstance(x, SpatRaster):
        raise TypeError("tile_apply: 'x' must be a SpatRaster")
    # SpatRaster.hasWindow() returns one bool per source (a vector); the
    # raster has a window when *any* source has one set.
    if any(x.hasWindow()):
        raise ValueError(
            "tile_apply: 'x' already has a window set; "
            "remove it first with remove_window(x)"
        )
    if not callable(fun):
        raise TypeError("tile_apply: 'fun' must be callable")

    if tiles is not None and buffer > 0:
        warnings.warn(
            "tile_apply: 'buffer' is only used when 'tiles' is None; ignoring it",
            stacklevel=2,
        )
        buffer = 0

    ncores = max(int(cores), 1)
    exts = _tile_apply_extents(x, tiles, cores=ncores, buffer=int(buffer))
    ntiles = len(exts)
    if ntiles == 0:
        raise ValueError("tile_apply: no tiles to process")

    tdir = tempfile.mkdtemp(prefix="tile_apply_")
    tile_files = [os.path.join(tdir, f"tile_{i:05d}.tif") for i in range(ntiles)]
    keep_tiles = False
    src_tmp: Optional[str] = None
    src_tmp_created = False

    try:
        if ncores > 1:
            src_tmp, src_tmp_created = _ensure_file_backed(x)

            fun_blob = _serialize_fun(fun)
            kw_blob = _serialize_fun(kwargs) if kwargs else b""
            jobs = [
                (
                    src_tmp,
                    p["outer"],
                    p["inner"],
                    tile_files[i],
                    fun_blob,
                    kw_blob,
                    datatype,
                )
                for i, p in enumerate(exts)
            ]

            from concurrent.futures import ProcessPoolExecutor
            out_files: List[str] = [None] * ntiles  # type: ignore
            with ProcessPoolExecutor(max_workers=ncores) as ex:
                for i, fpath in enumerate(ex.map(_tile_worker, jobs)):
                    out_files[i] = fpath
        else:
            # sequential path - same disk-streaming contract as the workers
            from .generics import crop as _crop
            from .window import set_window
            from .write import write_raster

            out_files = []
            for i, p in enumerate(exts):
                y = x.deepcopy()
                e = SpatExtent(*p["outer"])
                y = set_window(y, e)
                r = fun(y, **kwargs)
                if not isinstance(r, SpatRaster):
                    raise RuntimeError(
                        "tile_apply: 'fun' must return a SpatRaster for every tile"
                    )
                if p["outer"] != p["inner"]:
                    ie = SpatExtent(*p["inner"])
                    r = _crop(r, ie, snap="near")
                if datatype:
                    write_raster(r, tile_files[i], overwrite=True, datatype=datatype)
                else:
                    write_raster(r, tile_files[i], overwrite=True)
                out_files.append(tile_files[i])
                del r, y

        # ---- Single-tile shortcut: just promote the one tile file ----
        from .rast import rast as _rast
        if len(out_files) == 1:
            if filename:
                if os.path.exists(filename) and not overwrite:
                    raise FileExistsError(filename)
                shutil.copyfile(out_files[0], filename)
                return _rast(filename)
            # Move out of tdir so the result survives the cleanup below.
            fd, stable = tempfile.mkstemp(suffix=".tif", prefix="tile_apply_one_")
            os.close(fd)
            try:
                os.unlink(stable)
            except OSError:
                pass
            shutil.move(out_files[0], stable)
            return _rast(stable)

        # ---- Multi-tile assembly ----
        from .sprc import sprc

        if overlap_fun is None:
            # Cheap, lossless VRT assembly. Correct for non-overlapping
            # tiles (the auto-sized default); for overlapping tiles GDAL
            # keeps the value of the last tile drawn, so use overlap_fun
            # when tiles were built with get_tile_extents(buffer=).
            vrtfile = os.path.join(tdir, "all.vrt")
            rasters = [_rast(f) for f in out_files]
            rc = sprc(rasters)
            # SprcCollection.make_vrt() returns the VRT filename (mirrors
            # R: vrt() then rast(file)).
            vrt_path = rc.make_vrt(filename=vrtfile, overwrite=True)
            v = _rast(str(vrt_path))
            if filename:
                from .write import write_raster
                out = write_raster(v, filename, overwrite=overwrite, datatype=datatype)
                return out
            keep_tiles = True
            return v

        # Mosaic-based assembly handles overlap with overlap_fun.
        from .merge import mosaic
        rasters = [_rast(f) for f in out_files]
        return mosaic(*rasters, fun=overlap_fun, filename=filename, overwrite=overwrite)
    finally:
        if not keep_tiles:
            shutil.rmtree(tdir, ignore_errors=True)
        if src_tmp is not None and src_tmp_created:
            try:
                os.unlink(src_tmp)
            except OSError:
                pass
