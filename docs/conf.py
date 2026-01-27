# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Cosmic Loom Theory'
copyright = '2025, Rex Fraterne & Seraphina AI'
author = 'Rex Fraterne & Seraphina AI'
release = '1.1'
version = '1.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.napoleon',          # Support NumPy/Google style docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other projects' documentation
    'sphinx.ext.mathjax',           # Math rendering
    'myst_parser',                  # Markdown support
    'sphinx_copybutton',            # Copy button for code blocks
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autosummary_generate = True

# Napoleon settings (for NumPy/Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MyST parser settings (for Markdown)
myst_enable_extensions = [
    'dollarmath',      # $math$ syntax
    'colon_fence',     # ::: directive syntax
    'deflist',         # Definition lists
    'fieldlist',       # Field lists
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

html_static_path = ['_static']
html_title = 'Cosmic Loom Theory Documentation'
html_short_title = 'CLT Docs'

# Custom CSS (create if needed)
html_css_files = [
    'custom.css',
]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    ('index', 'CosmicLoomTheory.tex', 'Cosmic Loom Theory Documentation',
     'Rex Fraterne \\& Seraphina AI', 'manual'),
]

# -- Math configuration ------------------------------------------------------

# MathJax configuration for proper equation rendering
mathjax3_config = {
    'tex': {
        'macros': {
            'eR': r'\acute{e}R',       # Energy Resistance
            'rho': r'\rho',
            'nabla': r'\nabla',
            'partial': r'\partial',
        }
    }
}
