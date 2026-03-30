"""
sample.py — spatial sampling of SpatRaster and SpatExtent.
"""
from __future__ import annotations
from typing import Optional, Union
import math
import numpy as np

from ._terra import SpatRaster, SpatVector, SpatExtent, SpatOptions
from ._helpers import messages


def _opt() -> SpatOptions:
    return SpatOptions()


def _read_all_values(x: SpatRaster) -> np.ndarray:
    """Return (ncell, nlyr) float array, row-major order."""
    nr, nc, nl = x.nrow(), x.ncol(), x.nlyr()
    x.readStart()
    try:
        raw = x.readValues(0, nr, 0, nc)
    finally:
        x.readStop()
    arr = np.array(raw, dtype=float)
    # readValues returns values layer-by-layer (BSQ): lyr0[all cells], lyr1[all cells], …
    # reshape to (nlyr, ncell) then transpose to (ncell, nlyr)
    if nl > 1:
        arr = arr.reshape(nl, nr * nc).T
    else:
        arr = arr.reshape(nr * nc, 1)
    return arr


def _regular_nr_nc(x: SpatRaster, size: int) -> tuple[int, int]:
    """Compute (nr_sub, nc_sub) for a regular subgrid of ~size cells."""
    nr_r, nc_r = x.nrow(), x.ncol()
    nc_total = nr_r * nc_r
    if size >= nc_total:
        return nr_r, nc_r
    f = math.sqrt(size / nc_total)
    nr_sub = max(1, int(math.ceil(nr_r * f)))
    nc_sub = max(1, int(math.ceil(nc_r * f)))
    return nr_sub, nc_sub


def _regular_cells_0based(x: SpatRaster, nr_sub: int, nc_sub: int) -> np.ndarray:
    """0-based cell indices on a regular (nr_sub × nc_sub) subgrid of *x*."""
    nr_r, nc_r = x.nrow(), x.ncol()
    row_step = nr_r / nr_sub
    col_step = nc_r / nc_sub
    rows = np.array([int((i + 0.5) * row_step) for i in range(nr_sub)], dtype=int)
    cols = np.array([int((j + 0.5) * col_step) for j in range(nc_sub)], dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return (rr.ravel() * nc_r + cc.ravel()).astype(int)


def _build_df(
    x: SpatRaster,
    cell_idx: np.ndarray,
    val_arr: np.ndarray,
    want_cells: bool,
    want_xy: bool,
    want_values: bool,
) -> "pd.DataFrame":
    """Build a pandas DataFrame from 0-based *cell_idx* and extracted *val_arr*."""
    import pandas as pd

    d: dict = {}
    if want_cells:
        d["cell"] = cell_idx.astype(float)
    if want_xy:
        cell_floats = cell_idx.astype(float).tolist()
        xy_raw = x.xyFromCell(cell_floats)
        d["x"] = np.array(xy_raw[0], dtype=float)
        d["y"] = np.array(xy_raw[1], dtype=float)
    if want_values:
        lyr_names = list(x.names) if hasattr(x, "names") else []
        nl = val_arr.shape[1] if val_arr.ndim == 2 else 1
        if not lyr_names or len(lyr_names) != nl:
            lyr_names = [f"lyr{i + 1}" for i in range(nl)]
        for i, nm in enumerate(lyr_names):
            d[nm] = val_arr[:, i] if val_arr.ndim == 2 else val_arr
    return pd.DataFrame(d)


def spat_sample(
    x: Union[SpatRaster, SpatExtent],
    size: int,
    method: str = "random",
    *,
    replace: bool = False,
    na_rm: bool = False,
    as_raster: bool = False,
    as_points: bool = False,
    values: bool = True,
    cells: bool = False,
    xy: bool = False,
    lonlat: Optional[bool] = None,
    exact: bool = False,
    warn: bool = True,
) -> Union["pd.DataFrame", SpatVector, SpatRaster, np.ndarray]:
    """
    Draw a spatial sample from a SpatRaster or SpatExtent.

    Parameters
    ----------
    x : SpatRaster or SpatExtent
    size : int
        Number of samples.
    method : str
        ``"random"`` (default), ``"regular"``, or ``"stratified"``.
    replace : bool
        Sample with replacement.
    na_rm : bool
        Exclude NA cells.
    as_raster : bool
        Return sampled cells as a SpatRaster mask.
    as_points : bool
        Return sampled cells as a SpatVector of points.
    values : bool
        Include cell values in the output.
    cells : bool
        Include cell numbers.
    xy : bool
        Include x/y coordinates.
    lonlat : bool, optional
        Override CRS-based detection for geographic correction.
    exact : bool
        For regular sampling: trim to exactly *size* cells when the subgrid
        overshoots.
    warn : bool
        Warn when fewer samples than requested are available.

    Returns
    -------
    pandas.DataFrame, SpatVector, SpatRaster, or numpy.ndarray
    """
    size = max(1, int(round(size)))

    if isinstance(x, SpatExtent):
        return _sample_extent(x, size, method, lonlat, as_points, exact)

    method = method.lower()
    if method not in ("random", "regular", "stratified"):
        raise ValueError(
            f"method must be 'random', 'regular', or 'stratified'; got {method!r}"
        )

    # ── random ───────────────────────────────────────────────────────────────
    if method == "random":
        if as_raster:
            return messages(x.sampleRandomRaster(float(size), replace, 0), "spatSample")

        all_vals = _read_all_values(x)
        nc_total = all_vals.shape[0]

        if na_rm:
            valid_mask = ~np.any(np.isnan(all_vals), axis=1)
            valid_idx = np.where(valid_mask)[0]
        else:
            valid_idx = np.arange(nc_total, dtype=int)

        if len(valid_idx) == 0:
            try:
                import pandas as pd
                return pd.DataFrame()
            except ImportError:
                return np.empty((0, all_vals.shape[1]))

        n_avail = len(valid_idx)
        if size > n_avail and not replace:
            if warn:
                import warnings
                warnings.warn(
                    f"spat_sample: requested {size} but only {n_avail} valid cells available"
                )
            size = n_avail

        chosen = np.random.choice(n_avail, size=size, replace=replace)
        cell_idx = valid_idx[chosen]
        val_arr = all_vals[cell_idx]

        if as_points:
            return _cells_to_points(x, cell_idx)

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for spat_sample()")
        return _build_df(x, cell_idx, val_arr, cells, xy, values)

    # ── regular ───────────────────────────────────────────────────────────────
    if method == "regular":
        if as_raster:
            return messages(x.sampleRegularRaster(float(size), False), "spatSample")

        nr_sub, nc_sub = _regular_nr_nc(x, size)
        cell_idx = _regular_cells_0based(x, nr_sub, nc_sub)

        if exact and len(cell_idx) > size:
            cell_idx = cell_idx[:size]

        all_vals = _read_all_values(x)
        val_arr = all_vals[cell_idx]

        if as_points:
            return _cells_to_points(x, cell_idx)

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for spat_sample()")
        return _build_df(x, cell_idx, val_arr, cells, xy, values)

    # ── stratified ────────────────────────────────────────────────────────────
    if method == "stratified":
        all_vals = _read_all_values(x)
        nc_total = all_vals.shape[0]
        strata = all_vals[:, 0]
        unique_strata = np.unique(strata[np.isfinite(strata)])
        collected_cells = []
        for s in unique_strata:
            stratum_idx = np.where(strata == s)[0]
            n_pick = min(size, len(stratum_idx))
            chosen = np.random.choice(len(stratum_idx), size=n_pick, replace=False)
            collected_cells.append(stratum_idx[chosen])
        if collected_cells:
            cell_idx = np.concatenate(collected_cells)
        else:
            cell_idx = np.empty(0, dtype=int)
        val_arr = all_vals[cell_idx]

        if as_points:
            return _cells_to_points(x, cell_idx)

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for spat_sample()")
        return _build_df(x, cell_idx, val_arr, cells, xy, values)


def _cells_to_points(x: SpatRaster, cell_idx: np.ndarray) -> SpatVector:
    """Convert 0-based cell indices to a SpatVector of points."""
    cell_floats = cell_idx.astype(float).tolist()
    xy_raw = x.xyFromCell(cell_floats)
    coords = np.column_stack([
        np.array(xy_raw[0], dtype=float),
        np.array(xy_raw[1], dtype=float),
    ])
    from .vect import vect
    return vect(coords, type="points", crs=x.crs() if hasattr(x, "crs") else "")


def _sample_extent(
    x: SpatExtent,
    size: int,
    method: str,
    lonlat: Optional[bool],
    as_points: bool,
    exact: bool,
) -> Union[np.ndarray, SpatVector]:
    method = method.lower()
    if method not in ("random", "regular"):
        raise ValueError("method for SpatExtent must be 'random' or 'regular'")
    if lonlat is None:
        raise ValueError("lonlat must be specified when sampling from a SpatExtent")
    if method == "random":
        s = x.sampleRandom(size, lonlat, 0)
    else:
        s = x.sampleRegular(size, lonlat)
    arr = np.array(s, dtype=float).reshape(-1, 2)
    if as_points:
        from .vect import vect
        return vect(arr, type="points")
    return arr


# ---------------------------------------------------------------------------
# grid_sample — spatial thinning on a grid
# ---------------------------------------------------------------------------

def grid_sample(
    xy: Union[np.ndarray, SpatVector],
    r: SpatRaster,
    n: int = 1,
    chess: str = "",
) -> np.ndarray:
    """
    Thin point locations to at most *n* per raster cell.

    Parameters
    ----------
    xy : ndarray (n, 2) or SpatVector of points
        Input coordinates.
    r : SpatRaster
        Reference raster defining the grid.
    n : int
        Maximum number of points per cell.
    chess : str
        ``""`` (all cells), ``"white"`` or ``"black"`` (checkerboard
        sub-selection).

    Returns
    -------
    numpy.ndarray of indices (0-based) into *xy*.
    """
    if isinstance(xy, SpatVector):
        crds = np.array(xy.coordinates(), dtype=float).reshape(-1, 2)
    else:
        crds = np.asarray(xy, dtype=float)
        if crds.ndim == 1:
            crds = crds.reshape(1, -1)

    cell = r.cellFromXY(crds[:, 0].tolist(), crds[:, 1].tolist(), float("nan"))
    cell = np.array(cell, dtype=float)
    valid = ~np.isnan(cell)
    cell = cell.astype(int)

    if chess.lower() in ("white", "black"):
        nc = r.ncol()
        row_idx = cell // nc
        col_idx = cell % nc
        parity = (row_idx + col_idx) % 2
        keep_parity = 0 if chess.lower() == "white" else 1
        valid &= (parity == keep_parity)

    selected = []
    from collections import defaultdict
    cell_pts: dict = defaultdict(list)
    for i, (c, v) in enumerate(zip(cell, valid)):
        if v:
            cell_pts[c].append(i)

    rng = np.random.default_rng()
    for c, idxs in cell_pts.items():
        if len(idxs) <= n:
            selected.extend(idxs)
        else:
            chosen = rng.choice(idxs, n, replace=False)
            selected.extend(chosen.tolist())

    return np.array(sorted(selected), dtype=int)
