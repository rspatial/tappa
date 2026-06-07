"""
names.py — layer and variable names for SpatRaster and SpatVector.
"""
from __future__ import annotations
from typing import Any, List, Optional, Union

from ._terra import SpatRaster, SpatVector, SpatOptions

# Captured before methods.py monkey-patches SpatRaster/SpatVector.names.
_rast_names_desc = SpatRaster.names
_vect_names_desc = SpatVector.names


def _opt() -> SpatOptions:
    return SpatOptions()


def _cpp_layer_names(x: Union[SpatRaster, SpatVector]) -> List[str]:
    """Read layer/column names via the original C++ descriptor."""
    if isinstance(x, SpatRaster):
        desc = _rast_names_desc
    elif isinstance(x, SpatVector):
        desc = _vect_names_desc
    else:
        # Fallback for other C++ types (e.g. SpatVectorCollection) that still
        # have a raw .names attribute (not overridden by register_methods).
        raw = getattr(x, "names", None)
        if callable(raw):
            return list(raw())
        if raw is not None:
            return list(raw)
        raise TypeError(
            f"names: expected SpatRaster or SpatVector, got {type(x).__name__}"
        )
    if hasattr(desc, "__get__"):
        return list(desc.__get__(x, type(x)))
    return list(desc(x))


def _cpp_set_vect_names(x: SpatVector, value: List[str]) -> None:
    """Assign attribute column names via the original C++ descriptor."""
    if hasattr(_vect_names_desc, "__set__"):
        _vect_names_desc.__set__(x, [str(v) for v in value])
    else:
        raise RuntimeError("cannot set SpatVector names")


# ---------------------------------------------------------------------------
# SpatRaster layer names
# ---------------------------------------------------------------------------

def names(x: Union[SpatRaster, SpatVector]) -> List[str]:
    """Return layer or attribute column names — like R ``names()``."""
    if isinstance(x, SpatRaster):
        return _names_rast(x)
    if isinstance(x, SpatVector):
        return _names_vect(x)
    raise TypeError(
        f"names: expected SpatRaster or SpatVector, got {type(x).__name__}"
    )


def _names_rast(x: SpatRaster) -> List[str]:
    """Return the layer names of *x*."""
    return _cpp_layer_names(x)


def set_names(
    x: Union[SpatRaster, SpatVector],
    value: List[str],
    **kwargs: Any,
) -> Union[SpatRaster, SpatVector]:
    """Set layer or attribute column names."""
    if isinstance(x, SpatRaster):
        if kwargs.get("inplace"):
            _set_names_inplace(x, value, **kwargs)
            return x
        return _set_names_rast(x, value, **kwargs)
    if isinstance(x, SpatVector):
        return _set_names_vect(x, value)
    raise TypeError(
        f"set_names: expected SpatRaster or SpatVector, got {type(x).__name__}"
    )


def _set_names_rast(
    x: SpatRaster,
    value: List[str],
    index: Optional[List[int]] = None,
    validate: bool = False,
) -> SpatRaster:
    """
    Return a copy of *x* with new layer names.

    Parameters
    ----------
    x : SpatRaster
    value : list of str
        New names.  Length must equal nlyr(x) (or len(index) if provided).
    index : list of int, optional
        0-based layer indices to rename.  If None, rename all layers.
    validate : bool
        If True, sanitize names to be valid identifiers.

    Returns
    -------
    SpatRaster
    """
    xc = x.deepcopy() if hasattr(x, 'deepcopy') else x
    if index is None:
        index = list(range(xc.nlyr()))
    if len(value) != len(index):
        raise ValueError("length of value does not match length of index")
    current = _cpp_layer_names(xc)
    for i, v in zip(index, value):
        current[i] = str(v)
    if validate:
        import re
        seen: dict = {}
        valid = []
        for n in current:
            n = re.sub(r'[^A-Za-z0-9_.]', '.', n)
            if not n or n[0].isdigit():
                n = 'X' + n
            base = n
            cnt = seen.get(base, 0)
            if cnt:
                n = f"{base}.{cnt}"
            seen[base] = cnt + 1
            valid.append(n)
        current = valid
    if not xc.setNames(current, False):
        raise RuntimeError("cannot set these names")
    return xc


def _set_names_inplace(
    x: SpatRaster,
    value: List[str],
    index: Optional[List[int]] = None,
    validate: bool = False,
) -> None:
    """
    Modify layer names of *x* in-place (no copy).

    Parameters
    ----------
    x : SpatRaster
    value : list of str
    index : list of int, optional
        0-based layer indices.
    validate : bool
    """
    if index is None:
        index = list(range(x.nlyr()))
    current = _cpp_layer_names(x)
    for i, v in zip(index, value):
        current[i] = str(v)
    if not x.setNames(current, False):
        raise RuntimeError("cannot set these names")


# ---------------------------------------------------------------------------
# SpatVector attribute names
# ---------------------------------------------------------------------------

def _names_vect(x: SpatVector) -> List[str]:
    """Return the attribute column names of *x*."""
    return _cpp_layer_names(x)


def _set_names_vect(x: SpatVector, value: List[str]) -> SpatVector:
    """
    Return a copy of *x* with new attribute column names.

    Parameters
    ----------
    x : SpatVector
    value : list of str
        New column names.  Length must match ncol(x).

    Returns
    -------
    SpatVector
    """
    if len(value) != x.ncol():
        raise ValueError("incorrect number of names")
    xc = x.deepcopy()
    _cpp_set_vect_names(xc, [str(v) for v in value])
    return xc


# ---------------------------------------------------------------------------
# varnames — source/variable names embedded in a file
# ---------------------------------------------------------------------------

def varnames(x: SpatRaster) -> List[str]:
    """Return the variable (source) names of *x*."""
    return list(x.get_sourcenames())


def setVarnames(x: SpatRaster, value: List[str]) -> SpatRaster:
    """
    Return a copy of *x* with new variable names.

    Parameters
    ----------
    x : SpatRaster
    value : list of str

    Returns
    -------
    SpatRaster
    """
    xc = x.deepcopy() if hasattr(x, 'deepcopy') else x
    if not xc.set_sourcenames([str(v) for v in value]):
        raise RuntimeError("cannot set varnames")
    return xc


# ---------------------------------------------------------------------------
# longnames — human-readable long layer names
# ---------------------------------------------------------------------------

def longnames(x: SpatRaster) -> List[str]:
    """Return the long names of *x*."""
    return list(x.get_sourcenames_long())


def setLongnames(x: SpatRaster, value: List[str]) -> SpatRaster:
    """
    Return a copy of *x* with new long names.

    Parameters
    ----------
    x : SpatRaster
    value : list of str

    Returns
    -------
    SpatRaster
    """
    xc = x.deepcopy() if hasattr(x, 'deepcopy') else x
    if not xc.set_sourcenames_long([str(v) for v in value]):
        raise RuntimeError("cannot set longnames")
    return xc


# R-style alias
setNamesInplace = _set_names_inplace
