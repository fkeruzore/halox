# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "halox"
copyright = "2025, Florian Kéruzoré"
author = "Florian Kéruzoré"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinxcontrib.katex",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "sphinx_remove_toctrees",
    "sphinx_copybutton",
    # "jax_extensions",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "../../logo/logo_text.png"
html_favicon = "../../logo/logo.png"
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/fkeruzore/halox",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
}

# -- Options for myst -------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_render_image_options = {}
