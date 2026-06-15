# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import dotenv

sys.path.insert(0, os.path.abspath("../"))

from landscape_tools import __version__

project = "Landscape Tools"
package_name = "landscape_tools"
copyright = "2026, ColibriTD"
author = "ColibriTD"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_dark_mode",
    "sphinx_copybutton",
]
default_dark_mode = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": None,
    "undoc-members": None,
    "show-inheritance": None,
    # 'private-members': None,
    # 'special-members': '__init__',
    # 'member-order': 'bysource',
    # 'exclude-members': '__weakref__'
}
autodoc_member_order = "groupwise"
simplify_optional_unions = True
typehints_defaults = "comma"

dotenv.load_dotenv()
sphinx_github_changelog_token = os.getenv("SPHINX_GITHUB_CHANGELOG_TOKEN")
if sphinx_github_changelog_token is not None:
    extensions.append("sphinx_github_changelog")

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

html_context = {
    "display_github": True,
    "github_user": "ColibrITD-SAS",
    "github_repo": "landscape_tools",
    "github_version": "dev",
    "conf_py_path": "/docs/",
}

html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

html_short_title = "LT"

# html_logo = "resources/mpqp-logo-dark-theme.svg"
# html_favicon = "resources/favicon.ico"
html_use_smartypants = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_show_sphinx = False
htmlhelp_basename = package_name + "doc"

latex_documents = [
    (
        "index",
        package_name + ".tex",
        "Documentation of " + package_name,
        author,
        "manual",
    ),
]

man_pages = [("index", package_name, package_name + " documentation", [author], 1)]

texinfo_documents = [
    (
        "index",
        package_name,
        package_name + " documentation",
        author,
        package_name,
        project,
        "Miscellaneous",
    ),
]
