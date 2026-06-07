# Getting started

## Installation

`tappa` is a compiled extension that links against GDAL, GEOS, and PROJ. The
simplest way to get a working build depends on your platform.

### Windows (prebuilt wheels)

```
pip install tappa
```

Wheels are published on PyPI for CPython 3.9 – 3.12 on Windows x64. The wheels
bundle the native GDAL / GEOS / PROJ DLLs (built with vcpkg via `delvewheel`),
so no extra system libraries are required.

### Linux / macOS (build from source)

You need GDAL ≥ 3.4, GEOS ≥ 3.5, and PROJ ≥ 6, plus their development headers.

::::{tab-set}

:::{tab-item} Debian / Ubuntu
```bash
sudo apt-get install \
    libgdal-dev libgeos-dev libproj-dev \
    cmake pkg-config build-essential

pip install tappa
```
:::

:::{tab-item} conda / mamba
```bash
mamba install -c conda-forge gdal geos proj cmake pkg-config
pip install tappa
```
:::

:::{tab-item} Homebrew (macOS)
```bash
brew install gdal geos proj cmake pkg-config
pip install tappa
```
:::

::::

For a development checkout:

```bash
git clone https://github.com/rspatial/tappa
cd tappa
pip install -e .[plot]
```

## Hello world

```python
import tappa as pt

# Read a raster from disk
r = pt.rast("path/to/elevation.tif")
print(r)

# Crop to an extent and write the result
e  = pt.ext(640000, 660000, 4170000, 4190000)
r2 = pt.crop(r, e)
pt.write(r2, "elevation_cropped.tif")
```

## Two calling styles

Almost every operation can be called either as a function (matching the R
`terra` API) or as a method on the object:

```python
pt.crop(r, e)         #  R-like, functional
r.crop(e)             #  Pythonic, method

pt.aggregate(r, 2)    #  R-like
r.aggregate(2)        #  Pythonic
```

The two are equivalent.
