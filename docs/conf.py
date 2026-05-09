import hypersynchronization

project = "hypersynchronization"
author = "Maxime Lucas"
release = hypersynchronization.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/maximelucas/hypersynchronization",
    "show_toc_level": 2,
}

exclude_patterns = ["_build"]
