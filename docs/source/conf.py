"""Sphinx configuration for mathematicskit."""

import os
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))


def _release():
    """Installed distribution version, falling back to mathematicskit/__init__.py.

    The metadata lookup only succeeds when mathematicskit is installed (or a
    stale, gitignored egg-info happens to sit in the repo root), so a docs
    build from a fresh clone or a docs-only environment would otherwise die
    with PackageNotFoundError.
    """
    try:
        return _dist_version("mathematicskit")
    except PackageNotFoundError:
        init = Path(__file__).resolve().parents[2] / "mathematicskit" / "__init__.py"
        return re.search(r'^__version__\s*=\s*"([^"]+)"', init.read_text(), re.M).group(1)


project = "mathematicskit"
copyright = "2026, mathematicskit contributors"
author = "mathematicskit team"
release = _release()
del _release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Supports NumPy-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "jupyterlite_sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Single source of truth for the math subpackages. Every card grid
# (homepage, api/index, examples/index, history/index, and the per-subpackage
# hub pages) and the cross-link strip atop each api/examples/history page is
# generated from this table by _generate_subpackage_docs() below, instead of
# being hand-duplicated across five-plus RST files that used to drift out of
# sync with each other (different card ordering, stale blurbs, etc.) --
# ported structure-for-structure from physicskit's conf.py.
SUBPACKAGES = [
    {
        "name": "abstract_algebra",
        "category": "Foundations",
        "blurb": "Cyclic/permutation groups, subgroups and cosets, finite fields, and polynomial ring arithmetic.",
    },
    {
        "name": "calculus",
        "category": "Analysis",
        "blurb": "Numerical differentiation/integration, forward- and reverse-mode automatic differentiation, and Taylor series.",
    },
    {
        "name": "combinatorics",
        "category": "Foundations",
        "blurb": "Counting and generation, Pascal's triangle, integer partitions, inclusion-exclusion, and Stirling/Catalan/Bell numbers.",
    },
    {
        "name": "complex_analysis",
        "category": "Analysis",
        "blurb": "Contour integrals, Cauchy's theorem and formula, residues and the argument principle, conformal maps, and domain coloring.",
    },
    {
        "name": "fractals_chaos",
        "category": "Dynamics & Chaos",
        "blurb": "Lyapunov exponents, fractal dimension, Mandelbrot/Julia sets, iterated function systems, and cellular automata.",
    },
    {
        "name": "geometry",
        "category": "Geometry & Graphs",
        "blurb": "Convex hull, Delaunay triangulation/Voronoi diagrams, segment intersection/point-in-polygon, and the Frenet-Serret frame.",
    },
    {
        "name": "graph_theory",
        "category": "Geometry & Graphs",
        "blurb": "Shortest paths, minimum spanning trees, maximum flow/minimum cut, graph coloring, and spectral graph theory.",
    },
    {
        "name": "information_theory",
        "category": "Probability & Statistics",
        "blurb": "Entropy and mutual information, Huffman and arithmetic coding, channel capacity, and Hamming, convolutional and LDPC codes.",
    },
    {
        "name": "linalg",
        "category": "Linear Algebra & Optimization",
        "blurb": "LU/QR/Cholesky decompositions, eigenvalue algorithms, SVD, iterative Krylov solvers, and least-squares stability.",
    },
    {
        "name": "number_theory",
        "category": "Foundations",
        "blurb": "Modular arithmetic, primality testing, the Chinese Remainder Theorem, continued fractions, and Diophantine equations.",
    },
    {
        "name": "numerical_analysis",
        "category": "Analysis",
        "blurb": "Root finding with convergence-order verification, polynomial interpolation, and least-squares regression.",
    },
    {
        "name": "ode_dynamics",
        "category": "Dynamics & Chaos",
        "blurb": "Fixed-point stability, phase portraits, bifurcation normal forms, limit cycles, and Poincare sections.",
    },
    {
        "name": "optimization",
        "category": "Linear Algebra & Optimization",
        "blurb": "Gradient descent, nonlinear conjugate gradient, Newton/BFGS, constrained optimization, and linear programming.",
    },
    {
        "name": "pde",
        "category": "Analysis",
        "blurb": "Heat, wave, advection and Poisson/Laplace problems: the method of lines, Crank-Nicolson, CFL stability, and spectral methods.",
    },
    {
        "name": "probability",
        "category": "Probability & Statistics",
        "blurb": "Discrete/continuous distributions, Monte Carlo integration, limit theorems, and Markov chains.",
    },
    {
        "name": "special_functions",
        "category": "Analysis",
        "blurb": "Gamma/beta functions, Bessel functions, orthogonal polynomials, the FFT, and signal transforms: convolution, filters, Z/Laplace transforms, and wavelets.",
    },
    {
        "name": "statistics",
        "category": "Probability & Statistics",
        "blurb": "Descriptive statistics, hypothesis tests, confidence intervals, OLS regression, and bootstrap resampling.",
    },
    {
        "name": "topology",
        "category": "Geometry & Graphs",
        "blurb": "Simplicial homology and torsion, the fundamental group, classification of surfaces, fixed-point theorems, and persistent homology.",
    },
]

for _s in SUBPACKAGES:
    _s.setdefault("history_doc", f"{_s['name']}_breakthroughs")
    # Every subpackage's "Examples" link goes straight to its sphinx-gallery
    # index (:orphan: like every such index -- meant to be linked to
    # directly rather than placed in a toctree). Where a narrative tutorial
    # also exists, that gallery's examples/<name>/README.rst header links
    # out to it, so there's no separate examples/<name>.rst stub page whose
    # only job is forwarding to one or the other.
    _s["examples_doc"] = f"api/gallery/{_s['name']}/index"
del _s

_CATEGORY_ORDER = [
    "Foundations",
    "Analysis",
    "Linear Algebra & Optimization",
    "Geometry & Graphs",
    "Dynamics & Chaos",
    "Probability & Statistics",
]

# Subpackages with a sphinx-gallery-formatted examples/<name>/ directory.
_GALLERY_SUBPACKAGES = [s["name"] for s in SUBPACKAGES]


def _jupyterlite_install_cell(notebook_content, notebook_filename):
    """Prepend a ``%pip install`` cell so the Pyodide kernel fetches mathematicskit from PyPI."""
    notebook_content["cells"].insert(
        0,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%pip install mathematicskit"],
        },
    )


sphinx_gallery_conf = {
    "examples_dirs": [f"../../examples/{name}" for name in _GALLERY_SUBPACKAGES],
    "gallery_dirs": [f"api/gallery/{name}" for name in _GALLERY_SUBPACKAGES],
    "filename_pattern": r"/plot_",
    # Zip downloads of each gallery's .py/.ipynb files, so a course can take a
    # whole domain's examples at once.
    "download_all_examples": True,
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    # Lets ".. minigallery::" (used throughout docs/source/history/) resolve
    # fully-qualified object names in addition to the file paths/globs it
    # already handles, and is required for it to not warn about falling
    # back to file-path resolution on every single invocation.
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("mathematicskit",),
    # Embed each FuncAnimation as matplotlib's HTML/JavaScript player
    # (to_jshtml): PNG frames, no ffmpeg, the same player Jupyter and
    # JupyterLite show for `plt.rcParams["animation.html"] = "jshtml"`.
    "matplotlib_animations": (True, "jshtml"),
    # The examples end a cell with a bare `anim` so that notebooks display
    # the player; the scraper above already embeds it in the gallery page,
    # so don't also capture its repr.
    "ignore_repr_types": r"matplotlib\.animation\.",
    # "Launch JupyterLite" button: runs the notebook in the browser on
    # Pyodide, with no server. Numba cannot run there, so this relies on
    # numba being optional (mathematicskit/_jit.py).
    "jupyterlite": {
        "use_jupyter_lab": True,
        "notebook_modification_function": _jupyterlite_install_cell,
    },
}

# Many domains share common attribute names (e.g. "x", "n") across unrelated
# classes; autolinking to a single target for them isn't meaningful, so
# treat those as non-fatal rather than broken links.
suppress_warnings = ["ref.python", "config.cache"]

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/mathematicskit_logo_transparent.png"
html_theme_options = {
    "github_url": "https://github.com/cpoli/mathematicskit",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mathematicskit/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navigation_depth": 2,
    "logo": {
        "alt_text": "mathematicskit logo",
    },
}
html_static_path = ["_static"]


def _card(link, blurb, link_title):
    """One sphinx-design grid-item-card, indented for direct concatenation into a ``.. grid::`` block."""
    return f"   .. grid-item-card:: {link_title}\n      :link: {link}\n      :link-type: doc\n\n      {blurb}\n\n"


def _grid(cards):
    return ".. grid:: 1 2 3 3\n   :gutter: 2\n\n" + "".join(cards)


def _grouped_grid(link_fn):
    """A ``.. grid::`` per category (in _CATEGORY_ORDER), each preceded by a
    rubric heading, with ``link_fn(subpackage)`` giving each card's target.
    Shared by the homepage hub grid and the api/examples/history grids so
    they show the same category grouping instead of a flat alphabetical
    list.
    """
    parts = []
    for category in _CATEGORY_ORDER:
        members = [s for s in SUBPACKAGES if s["category"] == category]
        if not members:
            continue
        parts.append(f".. rubric:: {category}\n\n")
        parts.append(_grid([_card(link_fn(s), s["blurb"], f"mathematicskit.{s['name']}") for s in members]))
        parts.append("\n")
    return "".join(parts)


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _generate_subpackage_docs(app):
    """Generate the card grids, per-subpackage hub pages, and cross-link
    strips derived from SUBPACKAGES.

    Runs at "builder-inited", the same point sphinx-gallery uses to write
    its own generated RST, so the output exists before Sphinx reads any
    source file that ``.. include::``/``.. toctree::``\\ s it. Everything
    lands under _generated/ (gitignored, like api/gallery/) rather than
    being committed, since it's entirely derived from SUBPACKAGES above --
    there's nothing hand-authored in it to keep in version control.
    """
    out = Path(app.srcdir) / "_generated"

    # RST substitutions for numbers/facts derived from SUBPACKAGES, so prose
    # elsewhere (e.g. the homepage's "N domains" pitch) doesn't hardcode a
    # count that silently goes stale the next time a subpackage is added.
    _write(out / "vars.rst", f".. |num_subpackages| replace:: {len(SUBPACKAGES)}\n")

    # Category-grouped grids reused by index.rst, api/index.rst,
    # examples/index.rst, and history/index.rst, so those five-plus listings
    # of the same subpackages -- and their grouping -- come from one
    # place instead of being hand-copied (and drifting) independently.
    _write(out / "grid_api.rst", _grouped_grid(lambda s: f"/api/{s['name']}"))
    _write(out / "grid_examples.rst", _grouped_grid(lambda s: f"/{s['examples_doc']}"))
    _write(out / "grid_history.rst", _grouped_grid(lambda s: f"/history/{s['history_doc']}"))

    # Domain-first homepage grid: same grouping, linking to each
    # subpackage's own hub page rather than straight into a single
    # History/Examples/API silo.
    _write(out / "grid_hub.rst", _grouped_grid(lambda s: f"/_generated/subpackages/{s['name']}"))

    # Per-subpackage hub page: History, Examples, and API reference as
    # three equally-weighted doors into the same domain, so a reader who
    # wants "everything about linalg" doesn't have to browse three
    # separate site-wide sections to find linalg in each of them.
    hub_dir = out / "subpackages"
    for s in SUBPACKAGES:
        name = s["name"]
        title = f"mathematicskit.{name}"
        cards = _grid(
            [
                _card(
                    f"/history/{s['history_doc']}",
                    "The foundational breakthroughs behind this subpackage, linked to the implementation.",
                    "History",
                ),
                _card(
                    f"/{s['examples_doc']}",
                    "Runnable tutorials and the full example gallery.",
                    "Examples",
                ),
                _card(
                    f"/api/{name}",
                    "Every public class and function.",
                    "API reference",
                ),
            ]
        )
        _write(hub_dir / f"{name}.rst", f"{title}\n{'=' * len(title)}\n\n{s['blurb']}\n\n{cards}")

    # Cross-link strip included atop each hand-authored api/<name>.rst and
    # history/<name>_breakthroughs.rst page, so a reader who lands on just
    # one of them (e.g. from a search engine) can discover the others for
    # the same subpackage.
    nav_dir = out / "nav"
    for s in SUBPACKAGES:
        name = s["name"]
        links = [
            f":doc:`mathematicskit.{name} hub </_generated/subpackages/{name}>`",
            f":doc:`History </history/{s['history_doc']}>`",
            f":doc:`Examples </{s['examples_doc']}>`",
            f":doc:`API reference </api/{name}>`",
        ]
        _write(nav_dir / f"{name}.rst", f".. container:: subpkg-nav\n\n   {' · '.join(links)}\n")


def setup(app):
    app.connect("builder-inited", _generate_subpackage_docs)
