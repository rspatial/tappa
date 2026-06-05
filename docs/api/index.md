# API reference

The full public API of {mod}`tappa`. Every entry below is auto-generated from
the in-source NumPy-style docstrings; if something is missing from a function's
documentation, edit the docstring in `src/tappa/...py` and rebuild.

## Top-level package

```{eval-rst}
.. automodule:: tappa
   :no-members:
```

## Core C++ classes

These classes are exposed by the compiled extension (`tappa._terra`) and are
re-exported from the top-level package. Method-style operations like
`r.crop(e)` are attached at import time by `register_methods()`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   tappa.SpatRaster
   tappa.SpatVector
   tappa.SpatExtent
   tappa.SpatRasterStack
   tappa.SpatRasterCollection
   tappa.SpatVectorCollection
   tappa.SpatVectorProxy
   tappa.SpatDataFrame
   tappa.SpatFactor
   tappa.SpatCategories
   tappa.SpatTime_v
   tappa.SpatSRS
   tappa.SpatOptions
   tappa.SpatMessages
```

## Constructors

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   tappa.rast
   tappa.vect
   tappa.ext
   tappa.crs
```

## Submodules

The functional API is organised into thematic submodules. Each submodule is
also reachable directly (e.g. `tappa.geom.buffer_vect`).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:

   tappa.aggregate
   tappa.app
   tappa.arith
   tappa.cells
   tappa.coerce
   tappa.crosstab
   tappa.crs
   tappa.distance
   tappa.extent
   tappa.extract
   tappa.flow_accumulation
   tappa.focal
   tappa.freq
   tappa.generics
   tappa.geom
   tappa.init
   tappa.levels
   tappa.math
   tappa.merge
   tappa.methods
   tappa.names
   tappa.pitfinder
   tappa.plot
   tappa.rast
   tappa.rasterize
   tappa.relate
   tappa.sample
   tappa.sds
   tappa.show
   tappa.spatvec
   tappa.sprc
   tappa.stats
   tappa.subset
   tappa.tessellate
   tappa.tile_apply
   tappa.time
   tappa.values
   tappa.vect
   tappa.window
   tappa.write
   tappa.zonal
```
