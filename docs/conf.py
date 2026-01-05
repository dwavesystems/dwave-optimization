# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

# -- Project information -----------------------------------------------------

extensions = [
    # extensions provided by sphinx
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.ifconfig',

    # other extensions
    'breathe',
    'sphinx_design',
]

autosummary_generate = True

source_suffix = ['.rst']

master_doc = 'index'

language = 'en'

add_module_names = False

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', 'README.rst']

linkcheck_retries = 2
linkcheck_anchors = False
linkcheck_ignore = [r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    r'.clang-format',
                    r'setup.cfg',
                    ]

pygments_style = 'sphinx'

todo_include_todos = True

modindex_common_prefix = ['dwave.optimization.']

doctest_global_setup = """

import dwave.optimization

# print numpy numeric scalars without their type information,
# e.g. as 3.0 rather than np.float64(3.0).
import numpy
numpy.set_printoptions(legacy='1.25')
"""

autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    'np.typing.ArrayLike': 'numpy.typing.ArrayLike',
    'numpy.typing.ArrayLike': 'numpy.typing.ArrayLike',

    'ArraySymbolLike': 'ArraySymbol | numpy.typing.ArrayLike',
    'dwave.optimization.typing.ArraySymbolLike': 'ArraySymbol | numpy.typing.ArrayLike',
}

# -- Breathe --------------------------------------------------------------

breathe_projects = {
  'dwave-optimization': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'doxygen', 'xml'),
}

# see https://breathe.readthedocs.io/en/latest/readthedocs.html
if os.environ.get('READTHEDOCS', False):
    subprocess.call('make cpp', shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))

# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

intersphinx_mapping = {
    'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

rst_epilog = """
.. |array-like| replace:: array-like
.. _array-like: https://numpy.org/devdocs/glossary.html#term-array_like
"""
