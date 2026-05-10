import hypersynchronization
from sphinx_gallery.sorting import ExplicitOrder

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
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "subsection_order": ExplicitOrder([
        "../examples/states",
        "../examples/simulate",
        "../examples/visualize",
    ]),
}

html_title = "hypersynchronization"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {"text": "hypersynchronization"},
    "github_url": "https://github.com/maximelucas/hypersynchronization",
    "show_toc_level": 2,
}

exclude_patterns = ["_build"]

html_static_path = ["_static"]
