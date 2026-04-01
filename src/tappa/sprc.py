"""
sprc.py — SpatRasterCollection: a list of SpatRasters that may have
different extents, resolutions, or CRSs, like R ``terra::sprc()`` /
``SpatRasterCollection``.

The C++ backend is ``SpatRasterCollection`` (terra_lib/spatRasterMultiple.h).
"""
from __future__ import annotations

import warnings as _warnings
from typing import Any, Iterable, Iterator, List, Union

from ._helpers import messages as _msg, spatoptions
from ._terra import SpatOptions, SpatRaster, SpatRasterCollection as _SRC


# ── helpers ───────────────────────────────────────────────────────────────────

def _opt(filename: str = "", overwrite: bool = False) -> SpatOptions:
    return spatoptions(filename, overwrite)


def _check(ptr: _SRC, prefix: str) -> "SprcCollection":
    if ptr.has_error():
        raise RuntimeError(f"[{prefix}] {ptr.getError()}")
    for w in ptr.getWarnings():
        _warnings.warn(f"[{prefix}] {w}", stacklevel=3)
    return SprcCollection(ptr)


# ── class ─────────────────────────────────────────────────────────────────────

class SprcCollection:
    """
    A collection of :class:`SpatRaster` objects that may have different
    extents, resolutions, or CRSs, like R ``SpatRasterCollection``.

    Construct with :func:`sprc`.  Individual rasters are retrieved with
    ``[i]`` (0-based integer) or ``["name"]``.

    Examples
    --------
    >>> rc = sprc([r1, r2, r3])
    >>> len(rc)         # 3
    >>> rc[0]           # first raster
    >>> rc.names        # list of names
    >>> rc.merge()      # merge into a single SpatRaster
    >>> rc.mosaic()     # mosaic (mean of overlapping cells)
    """

    __slots__ = ("_ptr",)

    def __init__(self, ptr: _SRC) -> None:
        self._ptr = ptr

    # ── length / iteration ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return int(self._ptr.length())

    def __iter__(self) -> Iterator[SpatRaster]:
        for i in range(len(self)):
            yield self[i]

    # ── indexing ─────────────────────────────────────────────────────────────

    def __getitem__(self, key: Union[int, str, slice]) -> Union[SpatRaster, "SprcCollection"]:
        n = len(self)

        if isinstance(key, str):
            nms = list(self._ptr.names)
            if key not in nms:
                raise KeyError(f"raster {key!r} not in {nms}")
            key = nms.index(key)

        if isinstance(key, slice):
            indices = list(range(*key.indices(n)))
            new_ptr = _SRC()
            for i in indices:
                new_ptr.add(self._ptr.x[i], "")
            return _check(new_ptr, "sprc[slice]")

        idx = int(key)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"index {key} out of range for collection of length {n}")
        # .x is def_readonly → std::vector<SpatRaster>; index it directly
        return _msg(self._ptr.x[idx], "sprc[i]")

    # ── names ─────────────────────────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        return list(self._ptr.names)

    @names.setter
    def names(self, value: List[str]) -> None:
        self._ptr.names = [str(v) for v in value]

    # ── spatial metadata ──────────────────────────────────────────────────────

    def length(self) -> int:
        """Number of rasters in the collection."""
        return int(self._ptr.length())

    def dims(self) -> List[int]:
        """Dimensions (nrow, ncol) of the combined extent."""
        return [int(v) for v in self._ptr.dims()]

    @property
    def extent(self):
        """Bounding SpatExtent covering all rasters."""
        return self._ptr.extent

    # ── manipulation ──────────────────────────────────────────────────────────

    def merge(
        self,
        first: bool = False,
        na_rm: bool = False,
        filename: str = "",
        overwrite: bool = False,
    ) -> SpatRaster:
        """
        Merge all rasters into one, using the first non-NA value where
        rasters overlap — like R ``merge(SpatRasterCollection)``.

        Parameters
        ----------
        first : bool
            If True, the first raster takes priority everywhere.
        na_rm : bool
            Ignore NA values when selecting.
        """
        opt = _opt(filename, overwrite)
        return _msg(self._ptr.merge(first, na_rm, 1, "near", opt), "merge")

    def mosaic(
        self,
        fun: str = "mean",
        filename: str = "",
        overwrite: bool = False,
    ) -> SpatRaster:
        """
        Mosaic all rasters, aggregating overlapping values with *fun* —
        like R ``mosaic(SpatRasterCollection)``.

        Parameters
        ----------
        fun : str
            Aggregation function for overlap zones: ``"mean"`` (default),
            ``"min"``, ``"max"``, ``"sum"``, ``"median"``, ``"first"``,
            ``"last"``, or ``"blend"`` (distance-weighted feathering).
        """
        opt = _opt(filename, overwrite)
        return _msg(self._ptr.mosaic(fun, opt), "mosaic")

    def crop(
        self,
        extent: Any,
        snap: str = "near",
        expand: bool = False,
        filename: str = "",
        overwrite: bool = False,
    ) -> "SprcCollection":
        """Crop all rasters to *extent*."""
        from ._terra import SpatExtent
        e = extent if isinstance(extent, SpatExtent) else extent.extent
        opt = _opt(filename, overwrite)
        # use=[]: process all rasters
        return _check(self._ptr.crop(e, snap, expand, [], opt), "crop")

    def make_vrt(
        self,
        options: List[str] = None,
        reverse: bool = False,
        filename: str = "",
        overwrite: bool = False,
    ) -> SpatRaster:
        """Build a VRT mosaic from all rasters in the collection."""
        opt = _opt(filename, overwrite)
        return _msg(self._ptr.make_vrt(list(options or []), reverse, opt), "make_vrt")

    # ── add / remove ──────────────────────────────────────────────────────────

    def add(self, r: SpatRaster, name: str = "") -> None:
        """Append a SpatRaster to the collection (in-place)."""
        self._ptr.add(r, name)

    def erase(self, i: int) -> None:
        """Remove the raster at (0-based) index *i* (in-place)."""
        n = len(self)
        idx = int(i)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"index {i} out of range for collection of length {n}")
        self._ptr.erase(idx)

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self)
        nms = self.names
        lines = [f"SprcCollection with {n} raster(s)"]
        for i in range(min(n, 8)):
            nm = nms[i] if i < len(nms) else ""
            try:
                r = self[i]
                geo = f"{r.nrow()}×{r.ncol()}, {r.nlyr()} lyr(s)"
            except Exception:
                geo = "?"
            label = f" ({nm!r})" if nm else ""
            lines.append(f"  [{i}]{label}: {geo}")
        if n > 8:
            lines.append(f"  … {n - 8} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        return repr(self)


# ── factory function ──────────────────────────────────────────────────────────

def sprc(
    x: Union[None, str, SpatRaster, List[Any]] = None,
    /,
    *args: SpatRaster,
    ids: List[int] = None,
    opts: List[str] = None,
    noflip: bool = False,
    guess_crs: bool = True,
    domains: List[str] = None,
) -> SprcCollection:
    """
    Create a :class:`SprcCollection` — like R ``sprc()``.

    Parameters
    ----------
    x : None, str, SpatRaster, or list
        - ``None`` / no argument: empty collection.
        - ``str``: path to a multi-subdataset file.
        - ``SpatRaster``: first raster; additional rasters via ``*args``.
        - ``list``: list of :class:`SpatRaster` objects (may have different
          extents / CRS).
    *args : SpatRaster
        Additional rasters when *x* is a SpatRaster.
    ids : list of int, optional
        Sub-dataset IDs (1-based) to open from a file.
    opts : list of str, optional
        GDAL open options.
    noflip, guess_crs, domains
        Forwarded to the C++ constructor when opening from a file.

    Returns
    -------
    SprcCollection
    """
    # ── empty ─────────────────────────────────────────────────────────────────
    if x is None:
        return SprcCollection(_SRC())

    # ── from file path ────────────────────────────────────────────────────────
    if isinstance(x, str):
        _ids = [int(i) - 1 for i in (ids or [0])]
        use_ids = bool(ids)
        _opts = list(opts or [])
        _domains = list(domains or [""])
        ptr = _SRC(x, _ids, use_ids, _opts, noflip, guess_crs, _domains)
        return _check(ptr, "sprc")

    # ── from multiple file paths ──────────────────────────────────────────────
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        from .rast import rast as _rast
        ptr = _SRC()
        for path in x:
            nm = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
            ptr.add(_rast(path), nm)
        return _check(ptr, "sprc")

    # ── from SpatRaster + optional varargs ────────────────────────────────────
    if isinstance(x, SpatRaster):
        all_rasters = [x] + list(args)
        ptr = _SRC()
        for r in all_rasters:
            ptr.add(r, "")
        return _check(ptr, "sprc")

    # ── from list ─────────────────────────────────────────────────────────────
    if isinstance(x, (list, tuple)):
        ptr = _SRC()
        nms: List[str] = (
            list(x.names) if hasattr(x, "names") else [""] * len(x)
        )
        for i, item in enumerate(x):
            nm = nms[i] if i < len(nms) else ""
            if isinstance(item, SpatRaster):
                ptr.add(item, nm or "")
            elif isinstance(item, (SprcCollection,)):
                # flatten nested collection
                for j in range(len(item)):
                    sub_nm = item.names[j] if j < len(item.names) else ""
                    ptr.add(item._ptr.x[j], sub_nm)
            else:
                raise TypeError(
                    f"sprc: element {i} is {type(item).__name__!r}; "
                    "expected SpatRaster or SprcCollection"
                )
        rc = _check(ptr, "sprc")
        # propagate list names if the list had a dict-like interface
        if hasattr(x, "keys"):
            rc.names = list(x.keys())
        return rc

    raise TypeError(
        f"sprc: unsupported argument type {type(x).__name__!r}; "
        "pass None, str, SpatRaster, or list of SpatRasters"
    )
