# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime

project = "Pytorch-FINUFFT"
copyright = f"{datetime.date.today().year}, Simons Foundation"
author = "Brian Ward, Michael Eickenberg"


extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "texext.math_dollar",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# theme options
html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"]
html_show_sphinx = False

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/flatironinstitute/pytorch-finufft",
            "icon": "fab fa-github",
        },
    ],
    "use_edit_page_button": True,
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
}

html_context = {
    "github_user": "flatironinstitute",
    "github_repo": "pytorch-finufft",
    "github_version": "main",
    "doc_path": "docs",
}


autodoc_mock_imports = ["torch", "finufft", "cufinufft"]
autodoc_typehints = "signature"
napoleon_numpy_docstring = True

# these let us auto-generate links to related projects
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "finufft": ("https://finufft.readthedocs.io/en/latest/", None),
}


# Makes the copying behavior on code examples cleaner
# by removing things like In [10]: from the text to be copied
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# sphinx-gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "image_scrapers": ("matplotlib",),
    "filename_pattern": ".*\.py",  # execute all examples
    "only_warn_on_example_error": True,
    "reference_url": {
        "pytorch_finufft": None,  # local docs
    },
    "download_all_examples": False,
}
