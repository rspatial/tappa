"""
relate.py — spatial relationships between SpatVector geometries.
"""
from __future__ import annotations
from typing import List, Optional, Union
import numpy as np

from ._terra import SpatRaster, SpatVector, SpatExtent, SpatOptions
from ._helpers import messages


def _opt() -> SpatOptions:
    return SpatOptions()


_DE9IM_RELATIONS = {
    "intersects", "touches", "crosses", "overlaps",
    "within", "contains", "covers", "coveredBy",
    "disjoint", "equals",
}


def _to_vect(x) -> SpatVector:
    """Convert SpatRaster or SpatExtent to a SpatVector of polygons."""
    from .vect import vect as make_vect
    if isinstance(x, SpatVector):
        return x
    if isinstance(x, SpatRaster):
        return x.asPolygons(True, False, True, False, _opt())
    if isinstance(x, SpatExtent):
        return x.asPolygons()
    raise TypeError(f"Cannot convert {type(x)} to SpatVector")


def is_related(
    x: Union[SpatVector, SpatRaster, SpatExtent],
    y: Union[SpatVector, SpatRaster, SpatExtent],
    relation: str,
) -> np.ndarray:
    """
    Test a binary spatial relation between *x* and *y*.

    Parameters
    ----------
    x, y : SpatVector, SpatRaster, or SpatExtent
    relation : str
        One of: ``"intersects"``, ``"touches"``, ``"crosses"``,
        ``"overlaps"``, ``"within"``, ``"contains"``, ``"covers"``,
        ``"coveredBy"``, ``"disjoint"``, ``"equals"``.

    Returns
    -------
    numpy.ndarray of bool, one value per feature in *x*.
    """
    xv = _to_vect(x)
    yv = _to_vect(y)
    # C++ getPrepRelateFun has no handler for "equals"; use DE-9IM directly.
    if relation.lower() == "equals":
        relation = "T*F**FFF*"
    out = xv.is_related(yv, relation)
    messages(xv, "is_related")
    return np.array(out, dtype=bool)


def relate(
    x: SpatVector,
    y: SpatVector,
    relation: str,
    *,
    pairs: bool = False,
    na_rm: bool = True,
) -> Union[np.ndarray, "pd.DataFrame"]:
    """
    Compute a spatial relation matrix between *x* and *y*.

    Parameters
    ----------
    x : SpatVector
    y : SpatVector
    relation : str
        DE-9IM relation name or pattern.
    pairs : bool
        If True, return a 2-column array of feature index pairs (0-based)
        instead of a boolean matrix.
    na_rm : bool
        Remove NA results when *pairs=True*.

    Returns
    -------
    numpy.ndarray, shape (nrow(x), nrow(y)) of bool, or
    numpy.ndarray, shape (n, 2) of [id.x, id.y] (0-based) if pairs=True.
    """
    out = x.related_between(y, relation, na_rm)
    messages(x, "relate")
    if pairs:
        if len(out[0]) == 0:
            return np.empty((0, 2), dtype=int)
        m = np.column_stack([np.array(out[0], dtype=int),
                             np.array(out[1], dtype=int)])
        return m
    else:
        m = np.zeros((x.nrow(), y.nrow()), dtype=bool)
        if len(out[0]) > 0:
            rows = np.array(out[0], dtype=int)
            cols = np.array(out[1], dtype=int)
            m[rows, cols] = True
        return m


def relateSelf(
    x: SpatVector,
    relation: str = "intersects",
    *,
    symmetrical: bool = True,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Compute a spatial relation matrix within *x* (against itself).

    Parameters
    ----------
    x : SpatVector
    relation : str
    symmetrical : bool
        If True, return only the upper triangle (excluding diagonal).
    na_rm : bool

    Returns
    -------
    numpy.ndarray, shape (n, 2) of [id1, id2] (0-based) pairs.
    """
    n = x.nrow()
    out = x.related_between(x, relation, na_rm)
    messages(x, "relateSelf")
    if len(out[0]) == 0:
        return np.empty((0, 2), dtype=int)
    rows = np.array(out[0], dtype=int)
    cols = np.array(out[1], dtype=int)
    pairs = np.column_stack([rows, cols])
    # Remove self-relation
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if symmetrical:
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    return pairs


# ── adjacent / nearby (R: terra::adjacent, terra::nearby) ────────────────────

_ADJ_TYPES = {"intersects", "touches", "queen", "rook"}


def adjacent(
    x: SpatVector,
    type: str = "rook",
    pairs: bool = True,
    symmetrical: bool = False,
) -> np.ndarray:
    """
    Polygon adjacency matrix (R ``terra::adjacent`` for SpatVector).

    Parameters
    ----------
    x : SpatVector
        Polygons.
    type : str
        ``"rook"`` (shared edge), ``"queen"`` (shared edge or vertex; same as
        ``"touches"``), ``"intersects"``, or ``"touches"``.
    pairs : bool
        If True, return a 2-column ``[from, to]`` array of feature indices
        (0-based). Otherwise return an ``n x n`` boolean adjacency matrix
        with the diagonal set to False.
    symmetrical : bool
        If True (and ``pairs=True``) return each pair only once
        (``from < to``).

    Returns
    -------
    numpy.ndarray
    """
    t = type.lower()
    if t not in _ADJ_TYPES:
        raise ValueError(
            f"adjacent: type must be one of {sorted(_ADJ_TYPES)}; got {type!r}"
        )
    gtype = x.geomtype() if hasattr(x, "geomtype") and callable(x.geomtype) else None
    if isinstance(gtype, (list, tuple)):
        gtype = gtype[0] if gtype else None
    if gtype is not None and gtype != "polygons":
        raise ValueError("adjacent: x must contain polygons")

    a = relateSelf(x, t, symmetrical=False)
    n = x.nrow()
    if pairs:
        if symmetrical and len(a) > 0:
            a = a[a[:, 0] < a[:, 1]]
        return a
    m = np.zeros((n, n), dtype=bool)
    if len(a) > 0:
        m[a[:, 0], a[:, 1]] = True
    np.fill_diagonal(m, False)
    return m


def nearby(
    x: SpatVector,
    y: Optional[SpatVector] = None,
    distance: float = 0,
    k: int = 1,
    centroids: bool = True,
    symmetrical: bool = True,
    method: str = "geo",
    pairs: bool = False,
) -> np.ndarray:
    """
    Find neighbors of *x* (R ``terra::nearby``).

    Either ``distance > 0`` (return all pairs within that distance) or
    ``k >= 1`` (return the *k* nearest neighbors of each feature).

    Parameters
    ----------
    x : SpatVector
        Source features. If polygons and ``centroids=True``, distances are
        computed between centroids.
    y : SpatVector, optional
        Target features. If ``None``, uses *x* itself (within-set lookup).
    distance : float
        Distance threshold; pairs with distance ``<= distance`` are returned.
        Set to 0 to use *k* instead.
    k : int
        Number of nearest neighbors per feature. Ignored if ``distance > 0``.
    centroids : bool
        For polygons, use centroids when True.
    symmetrical : bool
        Within-set, distance-based mode only: return each pair once.
    method : str
        Distance method: ``"geo"``, ``"haversine"`` or ``"cosine"`` (same as
        ``terra::distance(SpatVector)``).
    pairs : bool
        For *k*-based mode, return ``[from, to]`` pairs instead of an
        ``n × (k+1)`` matrix. Ignored in distance-based mode (which always
        returns pairs).

    Returns
    -------
    numpy.ndarray
        For distance-based: ``[from, to]`` pairs (0-based).
        For *k*-based: an ``n × (k+1)`` matrix ``[id, k1, k2, …]``, or
        ``[from, to]`` pairs if ``pairs=True``.
    """
    from .distance import distanceVect, distanceVectSelf
    from .geom import centroids as _centroids

    k = int(round(k))
    if distance <= 0 and k < 1:
        raise ValueError("nearby: either 'distance' or 'k' must be a positive number")
    if method not in ("geo", "haversine", "cosine"):
        raise ValueError(
            f"nearby: method must be 'geo', 'haversine' or 'cosine'; got {method!r}"
        )

    def _maybe_centroid(v: SpatVector) -> SpatVector:
        gtype = v.geomtype() if callable(getattr(v, "geomtype", None)) else None
        if isinstance(gtype, (list, tuple)):
            gtype = gtype[0] if gtype else None
        if gtype == "polygons" and centroids:
            return _centroids(v)
        return v

    xc = _maybe_centroid(x)
    yc = _maybe_centroid(y) if y is not None else None

    if distance > 0:
        if yc is not None:
            d = distanceVect(xc, yc, method=method)
            r, c = np.indices(d.shape)
            mask = d <= distance
            return np.column_stack([r[mask], c[mask]])
        d = distanceVectSelf(xc, pairs=True, symmetrical=symmetrical, method=method)
        if d.size == 0:
            return np.empty((0, 2), dtype=int)
        return d[d[:, 2] <= distance][:, :2].astype(int)

    if yc is not None:
        d = distanceVect(xc, yc, method=method)
        kk = max(1, min(k, yc.nrow() - 1)) if yc.nrow() > 1 else 1
    else:
        d = distanceVectSelf(xc, method=method)  # n x n matrix
        np.fill_diagonal(d, np.nan)
        kk = max(1, min(k, xc.nrow() - 1))

    nb = np.argsort(d, axis=1, kind="stable")[:, :kk]
    out = np.column_stack([np.arange(d.shape[0], dtype=int), nb])
    if pairs:
        rows = np.repeat(out[:, 0], kk)
        cols = nb.ravel()
        return np.column_stack([rows, cols])
    return out
