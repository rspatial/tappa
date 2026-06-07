"""
Plotting for SpatRaster objects.

Provides :func:`plot` (single or multi-layer raster) and :func:`plot_rgb`
(composite colour image) using **matplotlib** as the rendering backend.

Quick start::

    import terra as pt
    import matplotlib.pyplot as plt

    r = pt.rast("elevation.tif")
    pt.plot(r)
    plt.show()
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._terra import SpatOptions, SpatRaster, SpatVector

__all__ = ["plot", "plot_rgb", "points", "lines", "polys", "text"]

# ── Default palette ──────────────────────────────────────────────────────────

def _default_palette(n: int = 100) -> List[str]:
    """
    Return the default tappa colour ramp.

    Mirrors R ``terra``'s ``.default.pal()`` — ``map.pal("viridis", 100)`` —
    so plots match what users see in the R package out of the box.
    Viridis is also colour-blind friendly and prints well in greyscale,
    and (importantly here) does not start at white, so the lowest interval
    of an interval plot is clearly visible against a white page.

    Args:
        n: Number of colours.

    Returns:
        List of hex colour strings.
    """
    import matplotlib as mpl
    import matplotlib.colors as mc
    cmap = mpl.colormaps["viridis"].resampled(n)
    return [mc.to_hex(cmap(i)) for i in range(n)]


# ── Value extraction ──────────────────────────────────────────────────────────

def _get_layer_array(r: SpatRaster, lyr: int) -> np.ndarray:
    """
    Read one layer (0-based index) as a 2-D float64 array (rows × cols).

    NA / Inf values become ``np.nan``.
    """
    opt = SpatOptions()
    flat = r.getValues(lyr, opt)
    arr = np.array(flat, dtype=np.float64).reshape(r.nrow(), r.ncol())
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _downsample_array(arr: np.ndarray, max_cells: int) -> np.ndarray:
    """
    Thin a 2-D array by a uniform stride so that ncell ≤ max_cells.

    Args:
        arr: 2-D numpy array (nrow × ncol).
        max_cells: Target maximum number of cells.

    Returns:
        Thinned array.
    """
    nr, nc = arr.shape
    total = nr * nc
    if total <= max_cells:
        return arr
    factor = math.ceil(math.sqrt(total / max_cells))
    return arr[::factor, ::factor]


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_colors: Sequence[str]) -> np.ndarray:
    """Convert a sequence of hex colour strings to an (N, 4) float array."""
    import matplotlib.colors as mc
    return np.array([mc.to_rgba(c) for c in hex_colors], dtype=np.float64)


def _breaks_equal_interval(
    values: np.ndarray, n_bins: int, range_vals: Tuple[float, float]
) -> np.ndarray:
    """Compute ``n_bins + 1`` equally-spaced break points over *range_vals*."""
    lo, hi = range_vals
    if lo == hi:
        return np.array([lo - 0.5, lo + 0.5])
    return np.linspace(lo, hi, n_bins + 1)


def _auto_digits(value_range: float) -> int:
    """Return appropriate decimal digit count for a legend given *value_range*."""
    if value_range == 0 or not math.isfinite(value_range):
        return 0
    return max(0, -math.floor(math.log10(value_range / 10)))


# ── Continuous pipeline ───────────────────────────────────────────────────────

def _continuous_image(
    arr: np.ndarray,
    palette: List[str],
    range_vals: Optional[Tuple[Optional[float], Optional[float]]] = None,
    fill_range: bool = False,
) -> Tuple[np.ndarray, Tuple[float, float], int]:
    """
    Map a continuous 2-D array to an RGBA image using *palette*.

    Args:
        arr: 2-D float array (rows × cols), NaN for no-data.
        palette: Ordered list of hex colour strings.
        range_vals: ``(vmin, vmax)`` clipping range.  ``None`` elements are
            filled from the data.
        fill_range: If True, clamp out-of-range values to the range endpoints
            rather than masking them.

    Returns:
        ``(rgba_image, (vmin, vmax), n_digits)`` where *rgba_image* has shape
        ``(rows, cols, 4)``.
    """
    import matplotlib.colors as mc

    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        rgba = np.zeros((*arr.shape, 4), dtype=np.float64)
        return rgba, (np.nan, np.nan), 0

    # Resolve data range
    data_min, data_max = float(np.nanmin(valid)), float(np.nanmax(valid))
    vmin: float = data_min
    vmax: float = data_max

    if range_vals is not None:
        lo, hi = range_vals
        if fill_range:
            if lo is not None and np.isfinite(lo):
                arr = np.where(arr < lo, lo, arr)
                vmin = lo
            else:
                vmin = data_min
            if hi is not None and np.isfinite(hi):
                arr = np.where(arr > hi, hi, arr)
                vmax = hi
            else:
                vmax = data_max
        else:
            vmin = lo if (lo is not None and np.isfinite(lo)) else data_min
            vmax = hi if (hi is not None and np.isfinite(hi)) else data_max

    n = len(palette)
    cmap = mc.ListedColormap(palette)

    if vmin == vmax:
        # Only one unique value — use the middle colour
        norm = mc.Normalize(vmin=vmin - 0.5, vmax=vmax + 0.5)
    else:
        norm = mc.Normalize(vmin=vmin, vmax=vmax)

    rgba = cmap(norm(np.ma.masked_invalid(arr)))
    # Restore transparency where data is NaN
    nan_mask = ~np.isfinite(arr)
    rgba[nan_mask, 3] = 0.0

    digits = _auto_digits(vmax - vmin)
    return rgba, (vmin, vmax), digits


# ── Classes pipeline ──────────────────────────────────────────────────────────

def _classes_image(
    arr: np.ndarray,
    palette: List[str],
    levels: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[float], List[str]]:
    """
    Map discrete class values in *arr* to colours from *palette*.

    Args:
        arr: 2-D float array.  Values not listed in *levels* become
            transparent.
        palette: List of hex colour strings.
        levels: Ordered list of numeric values to map.  If None, the unique
            non-NaN values in *arr* are used.

    Returns:
        ``(rgba_image, levels, hex_fill_colours)``
    """
    import matplotlib.colors as mc

    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        rgba = np.zeros((*arr.shape, 4), dtype=np.float64)
        return rgba, [], []

    if levels is None:
        levels = sorted(set(float(v) for v in valid))

    n_lvl = len(levels)
    n_col = len(palette)
    if n_lvl == 1:
        cols = [palette[-1]]
    elif n_lvl <= n_col:
        idx = np.round(np.linspace(0, n_col - 1, n_lvl)).astype(int)
        cols = [palette[i] for i in idx]
    else:
        cols = [palette[i % n_col] for i in range(n_lvl)]

    level_to_col = {lv: mc.to_rgba(c) for lv, c in zip(levels, cols)}

    rgba = np.zeros((*arr.shape, 4), dtype=np.float64)
    for lv, color in level_to_col.items():
        mask = arr == lv
        rgba[mask] = color

    return rgba, levels, cols


# ── Factor pipeline ───────────────────────────────────────────────────────────

def _cats_to_dict(cats_obj: Any) -> Tuple[List[float], List[str]]:
    """
    Extract ``(ids, labels)`` from a SpatCategories object.

    Returns a list of numeric IDs and a parallel list of label strings.
    SpatCategories.index is the 0-based column index of the active label
    column in df (see SpatRaster::getLabels in terra). The id is column 0;
    for a typical [id, label] table the label sits at index 1.
    """
    from ._helpers import _getSpatDF

    df = _getSpatDF(cats_obj.df)
    if df is None or df.empty or df.shape[1] < 2:
        return [], []
    try:
        ids = [float(v) for v in df.iloc[:, 0]]
        label_col = int(getattr(cats_obj, "index", 1))
        if label_col <= 0 or label_col >= df.shape[1]:
            label_col = min(1, df.shape[1] - 1)
        labels = [str(v) for v in df.iloc[:, label_col]]
    except Exception:
        return [], []
    return ids, labels


def _coltab_to_hex(coltab_obj: Any) -> Tuple[List[float], List[str]]:
    """
    Convert a SpatDataFrame colour table to ``(ids, hex_colours)``.

    The SpatDataFrame has columns: id, R, G, B, A (values 0–255).
    """
    from ._helpers import _getSpatDF

    df = _getSpatDF(coltab_obj)
    if df is None or df.empty or df.shape[1] < 5:
        return [], []
    try:
        ids = [float(v) for v in df.iloc[:, 0]]
        r = np.asarray(df.iloc[:, 1], dtype=np.uint8)
        g = np.asarray(df.iloc[:, 2], dtype=np.uint8)
        b = np.asarray(df.iloc[:, 3], dtype=np.uint8)
        a = np.asarray(df.iloc[:, 4], dtype=np.uint8)
        hexcols = [
            "#{:02x}{:02x}{:02x}{:02x}".format(ri, gi, bi, ai)
            for ri, gi, bi, ai in zip(r, g, b, a)
        ]
    except Exception:
        return [], []
    return ids, hexcols


def _factor_image(
    arr: np.ndarray,
    cats_obj: Any,
    coltab_obj: Optional[Any],
    palette: List[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Map a factor-coded raster to RGBA colours, using the category table.

    Args:
        arr: 2-D integer-valued float array.
        cats_obj: SpatCategories for this layer.
        coltab_obj: Optional colour-table SpatDataFrame for this layer.
        palette: Fallback colour palette.

    Returns:
        ``(rgba_image, legend_labels, hex_fill_colours)``
    """
    import matplotlib.colors as mc

    ids, labels = _cats_to_dict(cats_obj)
    if not ids:
        return np.zeros((*arr.shape, 4)), [], []

    if coltab_obj is not None:
        ct_ids, ct_hex = _coltab_to_hex(coltab_obj)
        id_to_color = {ct_ids[i]: ct_hex[i] for i in range(len(ct_ids))}
    else:
        n_lvl = len(ids)
        n_col = len(palette)
        if n_lvl <= n_col:
            idxs = np.round(np.linspace(0, n_col - 1, n_lvl)).astype(int)
            selected = [palette[i] for i in idxs]
        else:
            selected = [palette[i % n_col] for i in range(n_lvl)]
        id_to_color = {ids[i]: selected[i] for i in range(len(ids))}

    fill_cols = [id_to_color.get(iid, "#ffffff00") for iid in ids]

    rgba = np.zeros((*arr.shape, 4), dtype=np.float64)
    for iid, color in id_to_color.items():
        mask = arr == float(iid)
        rgba[mask] = mc.to_rgba(color)

    return rgba, labels, fill_cols


# ── Interval pipeline ─────────────────────────────────────────────────────────

def _interval_image(
    arr: np.ndarray,
    breaks: Union[np.ndarray, List[float]],
    palette: List[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Map values to colours according to user-defined break intervals.

    Each colour in *palette* is assigned to the half-open interval
    ``[breaks[i], breaks[i+1])``, matching R's ``cut(..., right=FALSE)``.

    Args:
        arr: 2-D float array.
        breaks: Monotonically increasing break points.  ``len(breaks)`` must
            equal ``len(palette) + 1``.
        palette: One colour per interval.

    Returns:
        ``(rgba_image, interval_labels, hex_fill_colours)``
    """
    import matplotlib.colors as mc

    breaks = np.asarray(breaks, dtype=np.float64)
    n_bins = len(breaks) - 1
    # Resample the (typically 255-entry) palette across n_bins so each
    # interval gets a distinct colour spanning the full ramp. R's
    # ``terra::plot(type="interval")`` does the same — slicing the first
    # *n_bins* colours collapses to one corner of the ramp (e.g. all light
    # grey for the default ``terrain_r``).
    if len(palette) >= n_bins:
        idx = np.linspace(0, len(palette) - 1, n_bins).round().astype(int)
        cols = [palette[int(i)] for i in idx]
    else:
        cols = list(palette) + [palette[-1]] * (n_bins - len(palette))

    bin_idx = np.digitize(arr, breaks[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rgba = np.zeros((*arr.shape, 4), dtype=np.float64)
    for i, c in enumerate(cols):
        mask = (bin_idx == i) & np.isfinite(arr)
        rgba[mask] = mc.to_rgba(c)

    labels = [f"{breaks[i]:.4g} – {breaks[i+1]:.4g}" for i in range(n_bins)]
    return rgba, labels, cols


# ── Colortable pipeline ───────────────────────────────────────────────────────

def _colortable_image(arr: np.ndarray, coltab_obj: Any) -> np.ndarray:
    """
    Map integer cell values to RGBA colours using an embedded colour table.

    Args:
        arr: 2-D integer-valued float array.
        coltab_obj: SpatDataFrame colour table (id, R, G, B, A columns,
            values 0–255).

    Returns:
        RGBA image array (rows × cols × 4).
    """
    import matplotlib.colors as mc

    ct_ids, ct_hex = _coltab_to_hex(coltab_obj)
    if not ct_ids:
        return np.zeros((*arr.shape, 4))

    id_to_color = {float(iid): mc.to_rgba(c) for iid, c in zip(ct_ids, ct_hex)}
    rgba = np.zeros((*arr.shape, 4), dtype=np.float64)
    for iid, color in id_to_color.items():
        mask = arr == iid
        rgba[mask] = color
    return rgba


# ── RGB pipeline ──────────────────────────────────────────────────────────────

def _stretch_band(band: np.ndarray, method: Optional[str]) -> np.ndarray:
    """
    Stretch band values to the [0, 1] range.

    Args:
        band: 2-D float array.
        method: ``"lin"`` for linear 2%–98% percentile stretch, ``"hist"``
            for histogram equalisation, or ``None`` for simple min-max.

    Returns:
        Array with values in [0, 1].
    """
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band)
    if method in ("lin", "linear"):
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
    if lo == hi:
        return np.zeros_like(band)
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(band)] = 0.0
    return out


def _rgb_image(
    r: SpatRaster,
    rgb_bands: Sequence[int],
    scale: float = 255.0,
    stretch: Optional[str] = None,
    na_color: str = "white",
) -> np.ndarray:
    """
    Compose an RGBA image from three (or four) raster bands.

    Args:
        r: SpatRaster with at least three layers.
        rgb_bands: 0-based layer indices for Red, Green, Blue (and optionally
            Alpha).
        scale: Maximum value used for normalisation (typically 255 for 8-bit
            data, or 65535 for 16-bit data).
        stretch: Optional contrast stretch.  ``"lin"`` applies a linear
            2%–98% percentile stretch; ``"hist"`` applies histogram
            equalisation.
        na_color: Hex or named colour used for cells where any band is NA.

    Returns:
        RGBA array (rows × cols × 4) with values in [0, 1].
    """
    import matplotlib.colors as mc

    bands = []
    for b in rgb_bands[:3]:
        arr = _get_layer_array(r, b)
        if stretch:
            arr = _stretch_band(arr, stretch)
        else:
            arr = np.clip(arr / scale, 0.0, 1.0)
            arr[~np.isfinite(arr)] = 0.0
        bands.append(arr)

    nr, nc = bands[0].shape
    rgba = np.ones((nr, nc, 4), dtype=np.float64)
    rgba[:, :, 0] = bands[0]
    rgba[:, :, 1] = bands[1]
    rgba[:, :, 2] = bands[2]

    if len(rgb_bands) >= 4:
        alpha_arr = _get_layer_array(r, rgb_bands[3])
        rgba[:, :, 3] = np.clip(alpha_arr / scale, 0.0, 1.0)
        rgba[:, :, 3][~np.isfinite(alpha_arr)] = 0.0

    # Mask cells where any band was NA
    na_mask = np.zeros((nr, nc), dtype=bool)
    for b in rgb_bands[:3]:
        arr = _get_layer_array(r, b)
        na_mask |= ~np.isfinite(arr)
    na_rgba = mc.to_rgba(na_color)
    rgba[na_mask] = na_rgba

    return rgba


# ── Legend helpers ────────────────────────────────────────────────────────────

def _add_continuous_legend(
    ax: Any,
    cmap: Any,
    norm: Any,
    n_ticks: int = 5,
    digits: int = 2,
    title: str = "",
) -> None:
    """Add a colourbar (continuous legend) to *ax*."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = ax.get_figure()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(title)
    ticks = np.linspace(norm.vmin, norm.vmax, n_ticks)
    cb.set_ticks(ticks)
    fmt = f"{{:.{digits}f}}"
    cb.set_ticklabels([fmt.format(t) for t in ticks])


def _add_class_legend(
    ax: Any,
    labels: List[str],
    colors: List[str],
    title: str = "",
    reverse: bool = False,
) -> None:
    """Add a categorical legend (patches) to *ax*."""
    import matplotlib.patches as mpatches

    if reverse:
        labels = labels[::-1]
        colors = colors[::-1]
    patches = [
        mpatches.Patch(facecolor=c, label=l)
        for c, l in zip(colors, labels)
    ]
    ax.legend(
        handles=patches,
        title=title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True,
    )


# ── Axes / title helpers ──────────────────────────────────────────────────────

def _aspect_for_extent(
    ext: Sequence[float], lonlat: bool
) -> float:
    """Return the y/x display aspect to use for a spatial plot.

    Mirrors :func:`terra::plot`'s rule:

    * lon/lat data → ``1 / cos(mean_latitude)`` so 1 deg of latitude renders
      the same length as 1 deg of longitude at the centre of the extent.
    * projected (planar) data → ``1`` (equal scaling on both axes).
    """
    if not lonlat:
        return 1.0
    _, _, ymin, ymax = ext[:4]
    mean_lat = 0.5 * (float(ymin) + float(ymax))
    # Clamp to avoid divide-by-zero / extreme stretching at the poles.
    mean_lat = max(min(mean_lat, 89.9), -89.9)
    return 1.0 / math.cos(math.radians(mean_lat))


def _setup_axes(
    ax: Any,
    ext: Sequence[float],
    axes: bool,
    lonlat: bool,
) -> float:
    """Configure axis ticks/labels and aspect ratio for a spatial plot.

    Returns the numeric y/x aspect that was applied so that callers can pass
    it on to ``imshow`` (which otherwise overrides ``ax.set_aspect``).
    """
    xmin, xmax, ymin, ymax = ext
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    asp = _aspect_for_extent(ext, lonlat)
    ax.set_aspect(asp)
    if not axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tick_params(labelsize=8)
    return asp


def _get_nrnc(nr: Optional[int], nc: Optional[int], nl: int) -> Tuple[int, int]:
    """Determine the subplot grid dimensions for *nl* layers."""
    if nc is not None and nr is not None:
        return int(nr), int(nc)
    if nc is not None:
        return math.ceil(nl / int(nc)), int(nc)
    if nr is not None:
        return int(nr), math.ceil(nl / int(nr))
    nc_auto = math.ceil(math.sqrt(nl))
    nr_auto = math.ceil(nl / nc_auto)
    return int(nr_auto), int(nc_auto)


# ── Infer plot type ───────────────────────────────────────────────────────────

def _infer_type(r: SpatRaster, lyr: int) -> str:
    """
    Decide the plot type for layer *lyr* (0-based), mirroring R's auto-detection.

    Returns one of ``"continuous"``, ``"factor"``, ``"colortable"``,
    ``"depends"``, or ``"rgb"``.
    """
    rgb = r.getRGB()
    if rgb and lyr in rgb:
        return "rgb"
    vt = r.valueType(False)
    has_cats = r.hasCategories()
    has_col = r.hasColors()
    if has_cats[lyr] and has_col[lyr]:
        return "factor"
    if has_col[lyr]:
        return "colortable"
    if has_cats[lyr]:
        return "factor"
    if vt[lyr] == 3:   # boolean
        return "factor"
    return "depends"


# ── Single-layer plot ─────────────────────────────────────────────────────────

def _plot_one_layer(
    r: SpatRaster,
    lyr: int,
    ax: Any,
    palette: List[str],
    type: str,
    zlim: Optional[Tuple[Optional[float], Optional[float]]],
    clamp: bool,
    breaks: Optional[Union[Sequence[float], np.ndarray]],
    levels: Optional[List[Any]],
    legend: bool,
    na_color: Optional[str],
    axes: bool,
    title: str,
    max_cell: int,
    alpha: float,
    smooth: bool,
) -> None:
    """
    Render a single raster layer into *ax*.

    This is the workhorse called by :func:`plot` for each panel.
    """
    import matplotlib.colors as mc

    ext_v = r.extent.vector                   # (xmin, xmax, ymin, ymax)
    lonlat = r.isLonLat()

    arr = _get_layer_array(r, lyr)
    arr = _downsample_array(arr, max_cell)

    interp = "bilinear" if smooth else "nearest"

    # ── detect type ──────────────────────────────────────────────────────────
    if type == "depends":
        valid_u = np.unique(arr[np.isfinite(arr)])
        if len(valid_u) < 9:
            type = "classes"
        else:
            type = "continuous"

    # ── colour pipeline ──────────────────────────────────────────────────────
    # Resolve default breaks for ``type="interval"`` so an empty ``breaks=``
    # doesn't fall through both branches and produce an empty plot. R
    # ``terra::plot`` similarly auto-derives breaks for ``type="interval"``.
    if type == "interval" and breaks is None:
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
            n_bins = 5
            breaks = (
                np.linspace(lo, hi, n_bins + 1)
                if hi > lo
                else np.array([lo - 0.5, lo + 0.5])
            )

    if type in ("continuous", "interval") and breaks is not None:
        breaks_arr = np.asarray(breaks, dtype=np.float64)
        rgba, labels, fill_cols = _interval_image(arr, breaks_arr, palette)
        asp = _setup_axes(ax, ext_v, axes, lonlat)
        ax.imshow(rgba, extent=ext_v, origin="upper",
                  interpolation=interp, aspect=asp, alpha=alpha)
        if legend:
            _add_class_legend(ax, labels, fill_cols, title=title)

    elif type == "continuous":
        rgba, (vmin, vmax), digits = _continuous_image(
            arr, palette, range_vals=zlim, fill_range=clamp
        )
        asp = _setup_axes(ax, ext_v, axes, lonlat)
        n = len(palette)
        cmap = mc.ListedColormap(palette)
        norm = mc.Normalize(
            vmin=vmin if np.isfinite(vmin) else 0,
            vmax=vmax if np.isfinite(vmax) else 1,
        )
        if not np.isnan(vmin) and not np.isnan(vmax):
            ax.imshow(rgba, extent=ext_v, origin="upper",
                      interpolation=interp, aspect=asp, alpha=alpha)
            if legend:
                _add_continuous_legend(ax, cmap, norm, digits=digits, title=title)
        else:
            ax.imshow(rgba, extent=ext_v, origin="upper",
                      interpolation=interp, aspect=asp)

    elif type == "classes":
        lv = [float(v) for v in levels] if levels is not None else None
        rgba, lv_used, fill_cols = _classes_image(arr, palette, levels=lv)
        asp = _setup_axes(ax, ext_v, axes, lonlat)
        ax.imshow(rgba, extent=ext_v, origin="upper",
                  interpolation=interp, aspect=asp, alpha=alpha)
        if legend:
            str_labels = [str(v) for v in lv_used]
            _add_class_legend(ax, str_labels, fill_cols, title=title)

    elif type == "factor":
        cats_list = r.getCategories()
        cats_obj = cats_list[lyr] if cats_list else None
        coltabs = r.getColors()
        coltab_obj = coltabs[lyr] if coltabs and r.hasColors()[lyr] else None
        if cats_obj is not None:
            rgba, leg_labels, fill_cols = _factor_image(
                arr, cats_obj, coltab_obj, palette
            )
        else:
            rgba, lv_used, fill_cols = _classes_image(arr, palette)
            leg_labels = [str(v) for v in lv_used]
        asp = _setup_axes(ax, ext_v, axes, lonlat)
        ax.imshow(rgba, extent=ext_v, origin="upper",
                  interpolation=interp, aspect=asp, alpha=alpha)
        if legend:
            _add_class_legend(ax, leg_labels, fill_cols, title=title)

    elif type == "colortable":
        coltabs = r.getColors()
        coltab_obj = coltabs[lyr] if coltabs else None
        if coltab_obj is not None:
            rgba = _colortable_image(arr, coltab_obj)
        else:
            rgba, _, _ = _classes_image(arr, palette)
        asp = _setup_axes(ax, ext_v, axes, lonlat)
        ax.imshow(rgba, extent=ext_v, origin="upper",
                  interpolation=interp, aspect=asp, alpha=alpha)

    elif type == "rgb":
        rgb_idxs = r.getRGB()
        if not rgb_idxs:
            rgb_idxs = [0, 1, 2]
        rgba = _rgb_image(r, rgb_idxs, stretch=None,
                          na_color=na_color or "white")
        asp = _setup_axes(ax, ext_v, axes, lonlat)
        ax.imshow(rgba, extent=ext_v, origin="upper",
                  interpolation=interp, aspect=asp)

    # ── NA colour overlay ────────────────────────────────────────────────────
    if na_color is not None and type not in ("rgb", "colortable"):
        _add_na_overlay(ax, arr, na_color, ext_v, interp, alpha, asp=ax.get_aspect())

    # ── title ────────────────────────────────────────────────────────────────
    if title:
        ax.set_title(title, fontsize=9)


def _add_na_overlay(
    ax: Any,
    arr: np.ndarray,
    na_color: str,
    ext: Sequence[float],
    interp: str,
    alpha: float,
    asp: Any = "auto",
) -> None:
    """Draw a solid overlay colour for NA cells.

    Inherits the axes aspect via *asp* so we don't undo the ratio set by
    :func:`_setup_axes` (matplotlib's ``imshow`` would otherwise clobber it).
    """
    import matplotlib.colors as mc
    import numpy as np

    na_rgba = mc.to_rgba(na_color)
    overlay = np.zeros((*arr.shape, 4), dtype=np.float64)
    mask = ~np.isfinite(arr)
    overlay[mask] = na_rgba
    ax.imshow(overlay, extent=ext, origin="upper",
              interpolation=interp, aspect=asp)


# ── Public API ────────────────────────────────────────────────────────────────

def plot(
    r: Union[SpatRaster, SpatVector],
    y: Union[int, List[int], str, List[str], None] = None,
    *,
    col: Optional[Union[List[str], str]] = None,
    type: Optional[str] = None,
    legend: bool = True,
    axes: bool = True,
    zlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    clamp: bool = False,
    levels: Optional[List[Any]] = None,
    breaks: Optional[Sequence[float]] = None,
    na_color: Optional[str] = "white",
    alpha: float = 1.0,
    smooth: bool = False,
    maxcell: int = 500_000,
    nc: Optional[int] = None,
    nr: Optional[int] = None,
    maxnl: int = 16,
    main: Optional[Union[str, List[str]]] = None,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot a SpatRaster or SpatVector.

    For :class:`SpatRaster` this draws one or more layers (possibly as a grid
    of subplots when multiple layers are selected). For :class:`SpatVector`
    a single panel is drawn with the appropriate geometry painter (points,
    lines, or filled polygons) and an aspect ratio chosen from the CRS, so
    callers don't need to assemble the figure themselves.

    Args:
        r: SpatRaster or SpatVector to plot.
        y: Layer selector.  Can be:

            * ``None`` — plot all layers (up to ``maxnl``).
            * ``int`` — 0-based layer index (Python convention; ``-1`` is last).
            * ``list[int]`` — multiple 0-based layer indices.
            * ``str`` or ``list[str]`` — layer name(s).

        col: Colour palette as a list of hex strings, or a matplotlib
            colormap name (e.g. ``"viridis"``).  Defaults to the terra
            terrain palette.
        type: Plot type.  One of ``"continuous"``, ``"classes"``,
            ``"factor"``, ``"interval"``, ``"colortable"``, ``"rgb"``.
            If None, the type is inferred from the data.
        legend: If True (default), draw a legend or colourbar.
        axes: If True (default), draw coordinate axis ticks and labels.
        zlim: ``(vmin, vmax)`` display range for continuous data.  ``None``
            elements are derived from the data.
        clamp: If True, values outside *zlim* are clamped to the range
            endpoints rather than shown as NA.
        levels: Explicit numeric values to use as class levels.
        breaks: Cut-point values for interval classification.  When supplied,
            *type* is automatically set to ``"interval"``.
        na_color: Colour for NA cells.  Use ``None`` to make them
            transparent.
        alpha: Overall opacity for the raster image (0–1).
        smooth: If True, apply bilinear interpolation when rendering
            (relevant only for display, not the data).
        maxcell: Maximum number of cells to render.  The raster is thinned
            by a uniform stride when ``ncell(r) > maxcell``.
        nc: Number of subplot columns (multi-layer only).
        nr: Number of subplot rows (multi-layer only).
        maxnl: Maximum number of layers to plot when ``y=None`` (default 16).
        main: Title string or list of strings (one per layer).
        ax: Existing matplotlib ``Axes`` to plot into (single-layer only).
        figsize: Figure size ``(width, height)`` in inches.
        **kwargs: Additional keyword arguments forwarded to
            ``matplotlib.pyplot.subplots``.

    Returns:
        * Single-layer: the ``matplotlib.axes.Axes`` used.
        * Multi-layer: 2-D ``numpy.ndarray`` of ``Axes``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import builtins

    if isinstance(r, SpatVector):
        return _plot_spatvector(
            r,
            y=y,
            col=col,
            legend=legend,
            axes=axes,
            alpha=alpha,
            main=main,
            ax=ax,
            figsize=figsize,
            **kwargs,
        )

    if not r.hasValues:
        warnings.warn("plot: SpatRaster has no cell values", stacklevel=2)

    nl_total = r.nlyr()
    from .names import _cpp_layer_names
    lyr_names = _cpp_layer_names(r)

    # ── resolve layer selection (0-based) ────────────────────────────────────
    if y is None:
        n_plot = min(nl_total, maxnl)
        lyrs_0based = list(builtins.range(n_plot))
    elif isinstance(y, str):
        lyrs_0based = [lyr_names.index(y)]
    elif isinstance(y, (list, tuple)) and y and isinstance(y[0], str):
        lyrs_0based = [lyr_names.index(n) for n in y]
    elif isinstance(y, (list, tuple)):
        lyrs_0based = [int(v) for v in y]
    else:
        lyrs_0based = [int(y)]

    def _norm_lyr(i: int) -> int:
        j = int(i)
        if j < 0:
            j = nl_total + j
        if j < 0 or j >= nl_total:
            raise IndexError(f"layer index {i!r} out of range for nlyr={nl_total}")
        return j

    lyrs_0based = [_norm_lyr(i) for i in lyrs_0based]

    # ── colour palette ────────────────────────────────────────────────────────
    if col is None:
        palette = _default_palette(255)
    elif isinstance(col, str):
        import matplotlib as mpl
        import matplotlib.colors as mc
        cmap_obj = mpl.colormaps[col].resampled(255)
        palette = [mc.to_hex(cmap_obj(i)) for i in builtins.range(255)]
    else:
        palette = list(col)

    # ── breaks imply interval type ────────────────────────────────────────────
    effective_type = type
    if breaks is not None and type is None:
        effective_type = "interval"

    # ── determine per-layer types ─────────────────────────────────────────────
    def _layer_type(lyr0: int) -> str:
        if effective_type is not None:
            return effective_type
        return _infer_type(r, lyr0)

    # ── single layer ──────────────────────────────────────────────────────────
    if len(lyrs_0based) == 1:
        lyr0 = lyrs_0based[0]
        ltype = _layer_type(lyr0)
        title_str = (main[0] if isinstance(main, (list, tuple)) else main) or lyr_names[lyr0]

        if ax is None:
            fig, ax_ = plt.subplots(1, 1, figsize=figsize or (6, 5), **kwargs)
        else:
            ax_ = ax

        _plot_one_layer(
            r, lyr0, ax_, palette, ltype,
            zlim=zlim, clamp=clamp, breaks=breaks,
            levels=levels, legend=legend, na_color=na_color,
            axes=axes, title=title_str,
            max_cell=maxcell, alpha=alpha, smooth=smooth,
        )
        return ax_

    # ── multi-layer ───────────────────────────────────────────────────────────
    n_plot = len(lyrs_0based)
    nrows, ncols = _get_nrnc(nr, nc, n_plot)

    if main is None:
        titles = [lyr_names[i] for i in lyrs_0based]
    elif isinstance(main, str):
        titles = [main] * n_plot
    else:
        titles = list(main) + [""] * max(0, n_plot - len(main))

    fig, axes_arr = plt.subplots(
        nrows, ncols,
        figsize=figsize or (4 * ncols, 3.5 * nrows),
        **kwargs,
    )

    axes_flat = np.array(axes_arr).flatten()

    for i, lyr0 in enumerate(lyrs_0based):
        ax_i = axes_flat[i]
        ltype = _layer_type(lyr0)
        _plot_one_layer(
            r, lyr0, ax_i, palette, ltype,
            zlim=zlim, clamp=clamp, breaks=breaks,
            levels=levels, legend=legend, na_color=na_color,
            axes=axes, title=titles[i],
            max_cell=maxcell // n_plot, alpha=alpha, smooth=smooth,
        )

    # Hide unused subplots
    for j in builtins.range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return np.array(axes_arr)


def plot_rgb(
    r: SpatRaster,
    red: int = 0,
    green: int = 1,
    blue: int = 2,
    alpha_band: Optional[int] = None,
    scale: float = 255.0,
    stretch: Optional[str] = None,
    smooth: bool = True,
    na_color: str = "white",
    axes: bool = False,
    ax: Optional[Any] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot a composite colour (RGB or RGBA) image from a multi-layer SpatRaster.

    Args:
        r: SpatRaster with at least three layers.
        red: 0-based layer index for the red channel.
        green: 0-based layer index for the green channel.
        blue: 0-based layer index for the blue channel.
        alpha_band: Optional 0-based layer index for the alpha (transparency)
            channel.
        scale: Maximum value for normalisation.  Use 255 for 8-bit data,
            65535 for 16-bit data, or 1 if bands are already in [0, 1].
        stretch: Contrast stretch method.  ``"lin"`` applies a linear
            2%–98% percentile stretch; ``None`` uses simple min-max
            normalisation to *scale*.
        smooth: If True (default), apply bilinear interpolation when
            rendering.
        na_color: Colour used for cells where any band is NA.
        axes: If True, draw coordinate axis ticks and labels.
        ax: Existing matplotlib ``Axes`` to plot into.
        figsize: Figure size ``(width, height)`` in inches.
        **kwargs: Additional keyword arguments forwarded to
            ``matplotlib.pyplot.subplots``.

    Returns:
        The ``matplotlib.axes.Axes`` used for the plot.
    """
    import matplotlib.pyplot as plt

    rgb_bands = [red, green, blue]
    if alpha_band is not None:
        rgb_bands.append(alpha_band)

    rgba = _rgb_image(r, rgb_bands, scale=scale, stretch=stretch, na_color=na_color)

    if ax is None:
        fig, ax_ = plt.subplots(1, 1, figsize=figsize or (6, 5), **kwargs)
    else:
        ax_ = ax

    ext_v = r.extent.vector
    lonlat = r.isLonLat()
    interp = "bilinear" if smooth else "nearest"

    asp = _setup_axes(ax_, ext_v, axes, lonlat)
    ax_.imshow(rgba, extent=ext_v, origin="upper",
               interpolation=interp, aspect=asp)
    return ax_


# ── SpatVector overlay helpers (R: points / lines / polys) ───────────────────

# Map R graphics aliases to matplotlib kwarg names. Only used when the matplotlib
# name has not already been supplied in **kw.
_R_TO_MPL_COMMON: Dict[str, str] = {
    "lwd": "linewidth",
    "lty": "linestyle",
    "alpha": "alpha",
}
_R_TO_MPL_POINT: Dict[str, str] = {
    "pch": "marker",
    "cex": "_cex",   # special-cased: pyplot.scatter wants ``s`` (area in pt^2)
    "col": "color",
    "bg": "facecolor",
    "fg": "edgecolor",
}
_R_TO_MPL_LINE: Dict[str, str] = {
    "col": "color",
}
_R_TO_MPL_POLY: Dict[str, str] = {
    "col": "facecolor",
    "border": "edgecolor",
    "fill": "facecolor",
}

# Most common R ``pch`` numeric codes mapped to matplotlib markers.
_PCH_TO_MARKER: Dict[Any, str] = {
    0: "s", 1: "o", 2: "^", 3: "+", 4: "x", 5: "D",
    6: "v", 7: "s", 8: "*", 15: "s", 16: "o", 17: "^",
    18: "D", 19: "o", 20: ".", 21: "o", 22: "s", 23: "D",
    24: "^", 25: "v",
}


def _resolve_overlay_axes(ax: Any) -> Any:
    """Return the Axes onto which an overlay (``points``/``lines``/``polys``)
    should draw.

    Always additive — these helpers never create new figures. If the user
    passes an explicit *ax* it is used as-is. Otherwise we draw onto the
    currently active Axes (``plt.gca()``); to plot a vector from scratch
    use :func:`plot` instead.
    """
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax
    return plt.gca()


def _reject_layout_kwargs(fname: str, kwargs: Dict[str, Any]) -> None:
    """Catch legacy ``add=``/``figsize=`` calls and redirect to :func:`plot`."""
    bad = [k for k in ("add", "figsize") if k in kwargs]
    if bad:
        raise TypeError(
            f"{fname}() does not accept {', '.join(repr(b) for b in bad)} — "
            "these helpers are pure overlays. Use pt.plot(v, ...) to draw a "
            "SpatVector from scratch (it sizes the figure and chooses the "
            "aspect ratio for you), then add overlays with "
            f"pt.{fname}(..., ax=...)."
        )


def _vector_extent(v: SpatVector) -> Optional[Sequence[float]]:
    """Return ``(xmin, xmax, ymin, ymax)`` for a SpatVector, or ``None``.

    Tolerates either method or property access patterns, since ``extent``
    is a method on :class:`SpatVector` (in contrast with the property on
    :class:`SpatRaster`).
    """
    try:
        e = v.extent() if callable(getattr(v, "extent", None)) else v.extent
        ev = e.vector if hasattr(e, "vector") else e
    except Exception:
        return None
    if ev is None or len(ev) < 4:
        return None
    return ev


def _translate_kwargs(
    kw: Dict[str, Any],
    mapping: Dict[str, str],
    *,
    is_point: bool = False,
) -> Dict[str, Any]:
    """Translate R graphics kwargs (``col``, ``lwd``, ...) into matplotlib names."""
    out: Dict[str, Any] = {}
    for k, v in kw.items():
        if k in mapping and mapping[k] not in out:
            out[mapping[k]] = v
        elif k in _R_TO_MPL_COMMON and _R_TO_MPL_COMMON[k] not in out:
            out[_R_TO_MPL_COMMON[k]] = v
        else:
            out[k] = v
    if is_point:
        # cex (relative size) -> matplotlib scatter ``s`` (point-area).
        # Default base size ~36 pt^2, matching ``pch=20``.
        cex = out.pop("_cex", out.pop("cex", None))
        if cex is not None:
            base = out.pop("s", 36)
            out["s"] = float(cex) * float(base)
        # Translate numeric pch codes.
        marker = out.get("marker")
        if isinstance(marker, (int, float)):
            out["marker"] = _PCH_TO_MARKER.get(int(marker), "o")
    return out


def _iter_geom_parts(v: SpatVector):
    """Yield (geom_id, part_id, xs, ys) for each ring of *v*.

    Resilient to the variable column layout of ``SpatVector.get_geometry()``:
    points are 4-column (geom, part, x, y), lines also 4-column, polygons add
    a fifth ``hole`` column. Multi-part geometries are split by (geom, part).
    """
    raw = list(v.get_geometry())
    if not raw:
        return
    n_cols = len(raw)
    # Columns are [geom, part, x, y, hole][:n_cols] (see spatvec.geom()).
    # We always need x and y at positions 2 and 3.
    if n_cols < 4:
        return
    geom_col = np.asarray(raw[0])
    part_col = np.asarray(raw[1])
    x_col = np.asarray(raw[2])
    y_col = np.asarray(raw[3])
    if x_col.size == 0:
        return
    # Split by (geom, part) preserving order.
    keys = np.column_stack([geom_col, part_col])
    # Find boundaries where (geom, part) changes from row to row.
    if keys.shape[0] == 1:
        yield (int(geom_col[0]), int(part_col[0]),
               x_col, y_col)
        return
    diff = np.any(keys[1:] != keys[:-1], axis=1)
    boundaries = np.flatnonzero(diff) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [keys.shape[0]]])
    for s, e in zip(starts, ends):
        yield (int(geom_col[s]), int(part_col[s]),
               x_col[s:e], y_col[s:e])


def _setup_vector_axes(
    ax: Any, v: SpatVector, axes: bool
) -> None:
    """Apply extent, aspect, and tick setup for a from-scratch SpatVector plot."""
    ev = _vector_extent(v)
    if ev is None:
        return
    xmin, xmax, ymin, ymax = ev
    # Pad by 4% of each side so geometries don't touch the axis frame.
    pad_x = 0.04 * (xmax - xmin) if xmax > xmin else 1.0
    pad_y = 0.04 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    try:
        lonlat = bool(v.isLonLat())
    except Exception:
        lonlat = False
    ax.set_aspect(_aspect_for_extent(ev, lonlat))
    if not axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tick_params(labelsize=8)


def _resolve_color_by_attr(
    v: SpatVector,
    y: Union[int, str, None],
    palette: Optional[Union[List[str], str]],
) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
    """Build per-feature colours when a SpatVector is to be coloured by attribute.

    Returns a triple ``(face_colors, legend_labels, legend_colors)``. All
    three are ``None`` when *y* is ``None`` (caller falls back to a single
    default colour). Numeric attributes are binned into 10 equal-interval
    classes; categorical attributes use one colour per unique value.
    """
    if y is None:
        return None, None, None
    import matplotlib.cm as cm
    import matplotlib.colors as mc

    try:
        from .values import vect_values
        df = vect_values(v)
    except Exception:
        return None, None, None
    cols = list(df.columns) if hasattr(df, "columns") else []
    col_name = y if isinstance(y, str) else (cols[int(y)] if cols else None)
    if col_name is None or col_name not in cols:
        return None, None, None
    values = df[col_name].tolist()

    if palette is None:
        pal = _default_palette(255)
    elif isinstance(palette, str):
        import matplotlib as mpl
        cmap_obj = mpl.colormaps[palette].resampled(255)
        pal = [mc.to_hex(cmap_obj(i)) for i in range(255)]
    else:
        pal = list(palette)

    numeric_vals: List[float] = []
    is_numeric = True
    for v_ in values:
        try:
            numeric_vals.append(float(v_))
        except (TypeError, ValueError):
            is_numeric = False
            break

    if is_numeric:
        arr = np.array(numeric_vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None, None, None
        n_bins = min(10, max(1, len(set(finite.tolist()))))
        breaks = np.linspace(finite.min(), finite.max(), n_bins + 1)
        idxs = np.clip(np.digitize(arr, breaks[:-1]) - 1, 0, n_bins - 1)
        palette_idxs = np.round(np.linspace(0, len(pal) - 1, n_bins)).astype(int)
        bin_colors = [pal[i] for i in palette_idxs]
        face = [bin_colors[i] for i in idxs]
        legend_labels = [
            f"{breaks[i]:.4g} – {breaks[i+1]:.4g}" for i in range(n_bins)
        ]
        return face, legend_labels, bin_colors

    cats = []
    for v_ in values:
        if v_ not in cats:
            cats.append(v_)
    palette_idxs = np.round(np.linspace(0, len(pal) - 1, len(cats))).astype(int)
    cat_colors = [pal[i] for i in palette_idxs]
    cat_to_color = {c: cat_colors[i] for i, c in enumerate(cats)}
    face = [cat_to_color[v_] for v_ in values]
    legend_labels = [str(c) for c in cats]
    return face, legend_labels, cat_colors


def _plot_spatvector(
    v: SpatVector,
    *,
    y: Union[int, str, None],
    col: Optional[Union[List[str], str]],
    legend: bool,
    axes: bool,
    alpha: float,
    main: Optional[Union[str, List[str]]],
    ax: Optional[Any],
    figsize: Optional[Tuple[float, float]],
    **kwargs: Any,
) -> Any:
    """Draw a :class:`SpatVector` from scratch (R ``terra::plot.SpatVector``).

    Geometry-aware: points → scatter, lines → polylines, polygons → filled
    polygons. When *y* identifies an attribute column the features are
    coloured by that attribute (numeric → equal-interval, otherwise →
    categorical). Caller-supplied style kwargs (``edgecolor``, ``linewidth``,
    ``border``, ``lwd``, ``lty``, ``s``, ``cex``, ``pch`` ...) are applied
    to the painter; per-feature fills come from *y* / *col*.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as _MplPolygon

    if ax is None:
        fig, ax_ = plt.subplots(1, 1, figsize=figsize or (6, 5))
    else:
        ax_ = ax

    gtype = v.type()
    face_colors, leg_labels, leg_colors = _resolve_color_by_attr(v, y, col)

    if gtype == "points":
        raw = list(v.get_geometry())
        if raw and len(raw) >= 4 and len(raw[2]) > 0:
            xs = np.asarray(raw[2], dtype=float)
            ys = np.asarray(raw[3], dtype=float)
            geom_ids = np.asarray(raw[0], dtype=int)
            kw = _translate_kwargs(dict(kwargs), _R_TO_MPL_POINT, is_point=True)
            kw.setdefault("alpha", alpha)
            kw.setdefault("s", 20)
            if face_colors is not None:
                kw["c"] = [face_colors[g - 1] for g in geom_ids]
                kw.pop("color", None)
            else:
                kw.setdefault("color", "black")
            ax_.scatter(xs, ys, **kw)

    elif gtype == "lines":
        kw_base = _translate_kwargs(dict(kwargs), _R_TO_MPL_LINE)
        kw_base.setdefault("alpha", alpha)
        kw_base.setdefault("linewidth", 1)
        for gid, _pid, xs, ys in _iter_geom_parts(v):
            if xs.size == 0:
                continue
            kw = dict(kw_base)
            if face_colors is not None:
                kw["color"] = face_colors[gid - 1]
            else:
                kw.setdefault("color", "black")
            ax_.plot(xs, ys, **kw)

    elif gtype == "polygons":
        kw_base = _translate_kwargs(dict(kwargs), _R_TO_MPL_POLY)
        kw_base.setdefault("edgecolor", "black")
        kw_base.setdefault("linewidth", 0.5)
        kw_base.setdefault("alpha", alpha)
        # Resolve the fill colour. R's ``terra::plot.SpatVector`` overloads
        # ``col``: with ``y`` it's a palette; without ``y`` a single string
        # becomes the fill, and a list/tuple is interpreted per-feature.
        # An explicit ``facecolor`` / ``border`` kwarg always wins.
        user_face = kw_base.pop("facecolor", None)
        if user_face is None and y is None and col is not None:
            user_face = col  # str → single colour; list → per-feature
        for gid, _pid, xs, ys in _iter_geom_parts(v):
            if xs.size == 0:
                continue
            verts = np.column_stack([xs, ys])
            if user_face is not None:
                face = (
                    user_face[gid - 1]
                    if isinstance(user_face, (list, tuple))
                    else user_face
                )
            elif face_colors is not None:
                face = face_colors[gid - 1]
            else:
                face = "lightgrey"
            ax_.add_patch(
                _MplPolygon(verts, closed=True, facecolor=face, **kw_base)
            )
    else:
        raise ValueError(f"plot: unsupported SpatVector type {gtype!r}")

    _setup_vector_axes(ax_, v, axes)

    if legend and leg_labels and leg_colors:
        title_str = (
            (main[0] if isinstance(main, (list, tuple)) else main) or
            (str(y) if y is not None else "")
        )
        _add_class_legend(ax_, leg_labels, leg_colors, title=title_str)
    elif main:
        ax_.set_title(
            main[0] if isinstance(main, (list, tuple)) else main, fontsize=9
        )
    return ax_


def points(
    x: SpatVector,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Add a points :class:`SpatVector` to an existing plot (R ``points``).

    This is a pure overlay: the points are drawn onto *ax* (or the current
    Axes when *ax* is omitted). To plot a SpatVector from scratch — with
    proper extent, aspect ratio, and optional fill colours — use
    :func:`plot`.

    R-style keyword aliases are honoured:

    * ``col`` → ``color``
    * ``cex`` → scaled into Matplotlib's scatter ``s`` (point area)
    * ``pch`` → ``marker`` (numeric pch codes are translated; strings pass through)
    * ``lwd`` → ``linewidth``

    Any other ``**kwargs`` are forwarded as-is to
    :func:`matplotlib.axes.Axes.scatter` so e.g. ``label=...``, ``zorder=...``
    work normally.
    """
    _reject_layout_kwargs("points", kwargs)
    if x.type() != "points":
        raise ValueError(
            f"points: SpatVector is of type {x.type()!r}, expected 'points'"
        )
    ax_ = _resolve_overlay_axes(ax)
    raw = list(x.get_geometry())
    if not raw or len(raw) < 4 or len(raw[2]) == 0:
        return ax_
    xs = np.asarray(raw[2], dtype=float)
    ys = np.asarray(raw[3], dtype=float)
    plot_kw = _translate_kwargs(dict(kwargs), _R_TO_MPL_POINT, is_point=True)
    ax_.scatter(xs, ys, **plot_kw)
    # ``scatter`` already updates the axes data limits, so autoscale_view()
    # extends the view to include the new points when the axes is fresh
    # (no-op when the user has explicitly fixed xlim / ylim, e.g. after
    # ``pt.plot(r, ax=ax)``).
    ax_.autoscale_view()
    return ax_


def lines(
    x: SpatVector,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Add a lines or polygons :class:`SpatVector` to an existing plot as line
    segments (R ``lines``).

    Pure overlay (no figure / aspect setup); use :func:`plot` to render a
    SpatVector from scratch. For polygon inputs each ring is drawn as a
    closed polyline (no fill); use :func:`polys` if you want filled
    polygons.

    R-style keyword aliases:

    * ``col`` → ``color``
    * ``lwd`` → ``linewidth``
    * ``lty`` → ``linestyle``

    Other ``**kwargs`` are forwarded to :meth:`matplotlib.axes.Axes.plot`.
    """
    _reject_layout_kwargs("lines", kwargs)
    gtype = x.type()
    if gtype not in ("lines", "polygons"):
        raise ValueError(
            f"lines: SpatVector is of type {gtype!r}, expected 'lines' or 'polygons'"
        )
    ax_ = _resolve_overlay_axes(ax)
    plot_kw = _translate_kwargs(dict(kwargs), _R_TO_MPL_LINE)
    label_used = False
    label = plot_kw.pop("label", None)
    for _gid, _pid, xs, ys in _iter_geom_parts(x):
        if xs.size == 0:
            continue
        if gtype == "polygons" and (xs[0] != xs[-1] or ys[0] != ys[-1]):
            xs = np.append(xs, xs[0])
            ys = np.append(ys, ys[0])
        kw = dict(plot_kw)
        if label is not None and not label_used:
            kw["label"] = label
            label_used = True
        ax_.plot(xs, ys, **kw)
    ax_.autoscale_view()
    return ax_


def polys(
    x: SpatVector,
    *,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Add a polygons :class:`SpatVector` to an existing plot (R ``polys``).

    Pure overlay onto *ax* (or the current Axes); use :func:`plot` to draw
    polygons from scratch.

    R-style keyword aliases:

    * ``col``    → ``facecolor``
    * ``border`` → ``edgecolor``
    * ``lwd``    → ``linewidth``
    * ``lty``    → ``linestyle``
    * ``alpha``  → ``alpha``

    Pass ``facecolor='none'`` (or ``col='none'``) for outlines only. Other
    ``**kwargs`` are forwarded to :class:`matplotlib.patches.Polygon`.
    """
    _reject_layout_kwargs("polys", kwargs)
    if x.type() != "polygons":
        raise ValueError(
            f"polys: SpatVector is of type {x.type()!r}, expected 'polygons'"
        )
    from matplotlib.patches import Polygon as _MplPolygon

    ax_ = _resolve_overlay_axes(ax)
    plot_kw = _translate_kwargs(dict(kwargs), _R_TO_MPL_POLY)
    plot_kw.setdefault("facecolor", "none")
    plot_kw.setdefault("edgecolor", "black")
    label = plot_kw.pop("label", None)
    label_used = False
    all_verts: List[np.ndarray] = []
    for _gid, _pid, xs, ys in _iter_geom_parts(x):
        if xs.size == 0:
            continue
        verts = np.column_stack([xs, ys])
        kw = dict(plot_kw)
        if label is not None and not label_used:
            kw["label"] = label
            label_used = True
        ax_.add_patch(_MplPolygon(verts, closed=True, **kw))
        all_verts.append(verts)
    # ``Axes.add_patch`` does not always extend ``dataLim`` for polygons,
    # so push the vertices in explicitly and then ``autoscale_view()`` so a
    # blank axes ends up showing the polygons. (No-op when the caller has
    # already set explicit xlim / ylim, e.g. after ``pt.plot(r, ax=ax)``.)
    if all_verts:
        ax_.update_datalim(np.vstack(all_verts))
        ax_.autoscale_view()
    return ax_


# R ``text()`` aliases. ``cex`` is special-cased: matplotlib's ``ax.text``
# wants ``fontsize`` (in points), so we multiply the rcParams default.
_R_TO_MPL_TEXT: Dict[str, str] = {
    "col": "color",
    "cex": "_cex",
    "family": "family",
    "srt": "rotation",
    "adj": "_adj",
}


def _resolve_text_labels(
    x: SpatVector,
    labels: Union[str, int, Sequence[Any], None],
) -> List[str]:
    """Resolve the *labels* argument of :func:`text` to a list of strings,
    one per feature, mirroring R ``terra::text``."""
    n = int(x.nrow())
    if labels is None:
        return [str(i + 1) for i in range(n)]
    if isinstance(labels, (str, int)):
        try:
            from .values import vect_values
            df = vect_values(x)
        except Exception:
            df = None
        cols = list(df.columns) if df is not None and hasattr(df, "columns") else []
        col_name: Optional[str] = None
        if isinstance(labels, str):
            if labels in cols:
                col_name = labels
        else:  # int → column index
            if cols and 0 <= int(labels) < len(cols):
                col_name = cols[int(labels)]
        if col_name is not None:
            return [str(v) for v in df[col_name].tolist()]
        # Fall through: treat scalar as a literal label repeated n times.
        return [str(labels)] * n
    return [str(v) for v in labels]


def text(
    x: SpatVector,
    labels: Union[str, int, Sequence[Any], None] = None,
    *,
    ax: Any = None,
    halo: bool = False,
    hc: str = "white",
    hw: float = 0.1,
    inside: bool = False,
    jitter: float = 0,
    **kwargs: Any,
) -> Any:
    """Add labels to features of a :class:`SpatVector` (R ``terra::text``).

    Pure overlay: labels are drawn at feature centroids onto *ax* (or the
    current Axes). Use :func:`plot` to draw the geometry first.

    Args:
        x: SpatVector (any geometry type). Labels are placed at centroids.
        labels: What to label features with. May be:

            * ``None`` — feature index (1-based, matching R).
            * column name (``str``) or column index (``int``) — values from
              that attribute column.
            * sequence — used as-is (must have ``nrow(x)`` entries).

        ax: Existing matplotlib ``Axes``. Defaults to the current Axes.
        halo: If True, draw a contrasting outline around each label.
        hc: Halo (outline) colour.
        hw: Halo (outline) line width in points.
        inside: Use ``point_on_surface``-style centroids that are guaranteed
            to fall inside the polygon. *Not yet implemented* — a regular
            geometric centroid is used; emits a warning when set.
        jitter: Random offset factor applied to each centroid (fraction of
            the data range), useful when many labels overlap. ``0`` (default)
            disables.
        **kwargs: Forwarded to :func:`matplotlib.axes.Axes.text`. R-style
            aliases are honoured: ``col``→``color``, ``cex``→``fontsize``
            (relative to the default), ``srt``→``rotation``, ``family``,
            ``adj``→``ha`` / ``va``.
    """
    _reject_layout_kwargs("text", kwargs)
    ax_ = _resolve_overlay_axes(ax)

    # Centroids of every feature.
    cv = x.centroid(False)
    raw = list(cv.get_geometry())
    if not raw or len(raw) < 4 or len(raw[2]) == 0:
        return ax_
    cx = np.asarray(raw[2], dtype=float)
    cy = np.asarray(raw[3], dtype=float)

    if inside:
        warnings.warn(
            "text(inside=True) is not yet implemented; using ordinary "
            "centroids. Some labels may fall outside their polygon.",
            stacklevel=2,
        )

    if jitter and jitter > 0:
        rng = np.random.default_rng()
        rx = (cx.max() - cx.min()) * float(jitter)
        ry = (cy.max() - cy.min()) * float(jitter)
        cx = cx + rng.uniform(-rx, rx, size=cx.shape)
        cy = cy + rng.uniform(-ry, ry, size=cy.shape)

    label_strs = _resolve_text_labels(x, labels)
    if len(label_strs) != cx.size:
        # ``centroid`` collapses multi-part features into a single point per
        # feature, so the lengths should match. If they don't (e.g. caller
        # passed a too-short labels sequence) recycle.
        label_strs = [label_strs[i % len(label_strs)] for i in range(cx.size)]

    # Translate R kwargs.
    plot_kw = _translate_kwargs(dict(kwargs), _R_TO_MPL_TEXT)
    cex = plot_kw.pop("_cex", None)
    if cex is not None:
        import matplotlib as _mpl
        base = float(plot_kw.pop("fontsize", _mpl.rcParams["font.size"]))
        plot_kw["fontsize"] = float(cex) * base
    adj = plot_kw.pop("_adj", None)
    if adj is not None:
        # R ``adj`` is (horiz, vert) in [0..1]; matplotlib uses ha/va keywords.
        if isinstance(adj, (list, tuple)) and len(adj) >= 2:
            ha_map = {0: "left", 0.5: "center", 1: "right"}
            va_map = {0: "bottom", 0.5: "center", 1: "top"}
            plot_kw.setdefault("ha", ha_map.get(adj[0], "center"))
            plot_kw.setdefault("va", va_map.get(adj[1], "center"))
    plot_kw.setdefault("ha", "center")
    plot_kw.setdefault("va", "center")

    # Optional halo via matplotlib path effects.
    path_effects = None
    if halo and hw and hw > 0:
        from matplotlib import patheffects as _pe
        path_effects = [
            _pe.Stroke(linewidth=float(hw) * 2, foreground=hc),
            _pe.Normal(),
        ]

    for i, lab in enumerate(label_strs):
        artist = ax_.text(float(cx[i]), float(cy[i]), lab, **plot_kw)
        if path_effects is not None:
            artist.set_path_effects(path_effects)

    return ax_
