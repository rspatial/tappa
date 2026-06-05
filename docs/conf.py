# Sphinx configuration for the tappa docs site.
#
# Builds with: sphinx-build -b html docs docs/_build/html
# CI deploys docs/_build/html to GitHub Pages (.github/workflows/docs.yml).

from __future__ import annotations

import os
import sys
from datetime import date

# Make the source tree importable so autodoc can introspect it. The
# extension module also has to be built first; see the docs CI workflow.
sys.path.insert(0, os.path.abspath("../src"))

import tappa  # noqa: E402  (after sys.path manipulation)

# -- Project info ------------------------------------------------------------

project = "tappa"
author = "Robert J. Hijmans and contributors"
copyright = f"{date.today().year}, {author}"

# Version comes from the installed package metadata so CI / local builds stay
# in sync without manually editing this file.
release = getattr(tappa, "__version__", "0.0.0")
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Allow both .rst and .md as source.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST settings.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Napoleon: NumPy-style docstrings (matches the style used in src/tappa/*.py).
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# Autosummary creates stub pages for everything we list in the API page.
autosummary_generate = True
autosummary_imported_members = True

# Autodoc defaults.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "undoc-members": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# pybind11 wraps things like `def crop(self, e):` as bound methods;
# don't try to follow signatures into the C extension where it's noisy.
autodoc_mock_imports = []

# Intersphinx: link to NumPy / Pandas / Python docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}


# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "tappa"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/rspatial/tappa/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Better default for code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
