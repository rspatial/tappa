"""
sds.py — SpatRasterDataset: a collection of SpatRasters that share the
same spatial grid, like R ``terra::sds()`` / ``SpatRasterDataset``.

The C++ backend is ``SpatRasterStack`` (terra_lib/spatRasterMultiple.h).
Sub-datasets are accessed by 0-based integer index or by name.
"""
from __future__ import annotations

import warnings as _warnings
from typing import Any, Iterable, Iterator, List, Optional, Union

from ._helpers import messages as _msg
from ._terra import SpatOptions, SpatRaster, SpatRasterStack


# ── helpers ───────────────────────────────────────────────────────────────────

def _opt() -> SpatOptions:
    return SpatOptions()


def _check(ptr: SpatRasterStack, prefix: str) -> "SpatRasterDataset":
    """Raise on C++ error, emit Python warning on C++ warning, wrap in class."""
    if ptr.has_error():
        raise RuntimeError(f"[{prefix}] {ptr.getError()}")
    for w in ptr.getWarnings():
        _warnings.warn(f"[{prefix}] {w}", stacklevel=3)
    return SpatRasterDataset(ptr)


# ── class ─────────────────────────────────────────────────────────────────────

class SpatRasterDataset:
    """
    A collection of SpatRasters sharing the same spatial grid geometry
    (extent, resolution, CRS), like R ``SpatRasterDataset``.

    Construct with :func:`sds`.  Index sub-datasets with ``[i]`` (0-based)
    or ``["name"]``.

    Examples
    --------
    >>> ds = sds([r1, r2])
    >>> len(ds)           # 2
    >>> ds[0]             # first sub-dataset as SpatRaster
    >>> ds["precip"]      # by name
    >>> ds.names          # list of sub-dataset names
    >>> ds.collapse()     # flatten to single multi-layer SpatRaster
    """

    __slots__ = ("_ptr",)

    def __init__(self, ptr: SpatRasterStack) -> None:
        self._ptr = ptr

    # ── length / iteration ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return int(self._ptr.nsds())

    def __iter__(self) -> Iterator[SpatRaster]:
        for i in range(len(self)):
            yield self[i]

    # ── indexing ─────────────────────────────────────────────────────────────

    def __getitem__(self, key: Union[int, str, slice]) -> Union[SpatRaster, "SpatRasterDataset"]:
        n = len(self)
        if isinstance(key, str):
            nms = list(self._ptr.names)
            if key not in nms:
                raise KeyError(f"sub-dataset {key!r} not in {nms}")
            key = nms.index(key)
        if isinstance(key, slice):
            indices = list(range(*key.indices(n)))
            return _check(self._ptr.subset(indices), "sds[slice]")
        idx = int(key)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"index {key} out of range for dataset of length {n}")
        r = self._ptr.getsds(idx)
        return _msg(r, "sds[i]")

    def __setitem__(self, i: int, value: SpatRaster) -> None:
        n = len(self)
        idx = int(i)
        if idx < 0:
            idx += n
        if idx == n:
            self._ptr.add(value, "", "", "", False)
        elif 0 <= idx < n:
            self._ptr.replace(idx, value, False)
        else:
            raise IndexError(f"index {i} out of range for dataset of length {n}")

    # ── names ─────────────────────────────────────────────────────────────────

    @property
    def names(self) -> List[str]:
        return list(self._ptr.names)

    @names.setter
    def names(self, value: List[str]) -> None:
        self._ptr.names = [str(v) for v in value]

    # ── spatial metadata ──────────────────────────────────────────────────────

    def nsds(self) -> int:
        """Number of sub-datasets."""
        return int(self._ptr.nsds())

    def nlyr(self) -> List[int]:
        """Number of layers per sub-dataset."""
        return [int(v) for v in self._ptr.nlyr()]

    def nrow(self) -> int:
        return int(self._ptr.nrow())

    def ncol(self) -> int:
        return int(self._ptr.ncol())

    def res(self) -> List[float]:
        return list(self._ptr.res())

    @property
    def extent(self):
        return self._ptr.ext()

    def crs(self, style: str = "wkt") -> str:
        """CRS as WKT (default), PROJ, or EPSG string."""
        return self._ptr.get_crs(style)

    def filenames(self) -> List[str]:
        return list(self._ptr.filenames())

    # ── manipulation ──────────────────────────────────────────────────────────

    def collapse(self) -> SpatRaster:
        """Flatten all sub-datasets into a single multi-layer SpatRaster."""
        return _msg(self._ptr.collapse(), "collapse")

    def subset(self, indices: Iterable[int]) -> "SpatRasterDataset":
        """Return a new dataset keeping only the given (0-based) indices."""
        return _check(self._ptr.subset([int(i) for i in indices]), "subset")

    def summary(self, fun: str = "mean", na_rm: bool = True) -> SpatRaster:
        """Per-cell summary statistic across all sub-datasets."""
        return _msg(self._ptr.summary(fun, na_rm, _opt()), "summary")

    def crop(self, extent: Any, snap: str = "near", expand: bool = False) -> "SpatRasterDataset":
        """Crop all sub-datasets to *extent*."""
        from ._terra import SpatExtent
        e = extent if isinstance(extent, SpatExtent) else extent.extent
        return _check(self._ptr.crop(e, snap, expand, _opt()), "crop")

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self)
        nms = self.names
        try:
            rows, cols = self.nrow(), self.ncol()
            lyrs = self.nlyr()
            geo = f"{rows}×{cols}"
        except Exception:
            geo = "?"
            lyrs = ["?"] * n
        lines = [f"SpatRasterDataset with {n} sub-dataset(s)"]
        for i in range(n):
            nm = nms[i] if i < len(nms) else f"[{i}]"
            lyr_s = str(lyrs[i]) if i < len(lyrs) else "?"
            lines.append(f"  [{i}] {nm!r}: {geo}, {lyr_s} lyr(s)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self._ptr.show()


# ── factory function ──────────────────────────────────────────────────────────

def sds(
    x: Union[None, str, SpatRaster, List[Any]] = None,
    /,
    *args: SpatRaster,
    ids: List[int] = None,
    opts: List[str] = None,
    noflip: bool = False,
    guess_crs: bool = True,
    domains: List[str] = None,
    md: Optional[bool] = None,
) -> SpatRasterDataset:
    """
    Create a :class:`SpatRasterDataset` — like R ``sds()``.

    Parameters
    ----------
    x : None, str, SpatRaster, or list
        - ``None`` / no argument: empty dataset.
        - ``str``: path to a multi-subdataset file (e.g. NetCDF).
        - ``SpatRaster``: first sub-dataset; additional rasters via ``*args``.
        - ``list``: list of :class:`SpatRaster` objects.
    *args : SpatRaster
        Additional rasters when *x* is a SpatRaster.
    ids : list of int, optional
        Sub-dataset IDs to open from a file (1-based, like R).
    opts : list of str, optional
        GDAL open options.
    noflip : bool
        Suppress automatic y-flip.
    guess_crs : bool
        Try to guess CRS from file metadata.
    domains : list of str, optional
        GDAL metadata domains.
    md : bool, optional
        Multidimensional mode for NetCDF files.  ``True`` keeps all
        dimensions as separate layers, ``False`` collapses them.
        ``None`` (default) lets terra decide.

    Returns
    -------
    SpatRasterDataset
    """
    if md is None:
        md_int = 2
    else:
        md_int = int(md)

    ptr = SpatRasterStack()

    if x is None:
        return SpatRasterDataset(ptr)

    # ── from file path ────────────────────────────────────────────────────────
    if isinstance(x, str):
        _ids = [int(i) - 1 for i in (ids or [0])]
        use_ids = bool(ids)
        _opts = list(opts or [])
        _domains = list(domains or [""])
        file_ptr = SpatRasterStack(
            x, _ids, use_ids, _opts, noflip, guess_crs, _domains, md_int
        )
        return _check(file_ptr, "sds")

    # ── from a list of strings (multiple files) ───────────────────────────────
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
        from .rast import rast as _rast
        rasters = [_rast(p) for p in x]
        basenames = [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0] for p in x]
        for r, nm in zip(rasters, basenames):
            ptr.add(r, nm, "", "", False)
        return _check(ptr, "sds")

    # ── from SpatRaster + optional varargs ────────────────────────────────────
    if isinstance(x, SpatRaster):
        all_rasters = [x] + list(args)
        for r in all_rasters:
            ptr.add(r, "", "", "", False)
        return _check(ptr, "sds")

    # ── from list of SpatRasters ──────────────────────────────────────────────
    if isinstance(x, (list, tuple)):
        nms: List[str] = getattr(x, "__names__", None) or (
            x.names if hasattr(x, "names") else [""] * len(x)
        )
        for i, item in enumerate(x):
            if isinstance(item, SpatRaster):
                nm = nms[i] if i < len(nms) else ""
                ptr.add(item, nm or "", "", "", False)
            elif isinstance(item, SpatRasterDataset):
                for j in range(len(item)):
                    sub = item[j]
                    sub_nm = item.names[j] if j < len(item.names) else ""
                    ptr.add(sub, sub_nm, "", "", False)
            else:
                raise TypeError(
                    f"sds: element {i} is {type(item).__name__!r}; "
                    "expected SpatRaster or SpatRasterDataset"
                )
        return _check(ptr, "sds")

    raise TypeError(
        f"sds: unsupported argument type {type(x).__name__!r}; "
        "pass None, str, SpatRaster, or list of SpatRasters"
    )
