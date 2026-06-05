# tappa

**tappa** is a Python interface to the [terra](https://github.com/rspatial/terra)
geospatial C++ library. It exposes the same core classes as the R package
(`SpatRaster`, `SpatVector`, `SpatExtent`, …) and a function-level API that
mirrors R `terra` so workflows port over with minimal renaming.

```python
import tappa as pt

r = pt.rast("elev.tif")
e = pt.ext(0, 1, 0, 1)
r2 = pt.crop(r, e)         # functional style (R-like)
r2 = r.crop(e)             # method style (Pythonic)
```

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
user-guide/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

GitHub repository <https://github.com/rspatial/tappa>
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
