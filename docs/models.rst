.. _optimization_models:

=========================================
Nonlinear Models: Construction and States
=========================================

The `dwave-optimization` package provides the
:class:`~dwave.optimization.model.Model` and
:class:`~dwave.optimization.model.States` classes to construct nonlinear models
and handle results, respectively. These models map to a
`directed acyclic graph <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_
constituted of the package's :ref:`optimization_symbols` classes.

For examples, see the :ref:`opt_index_examples_beginner` section.

.. currentmodule:: dwave.optimization

Models and States
=================

.. automodule:: dwave.optimization.model

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    Model
    States

.. _optimization_symbols:

Symbols
=======

Symbols are a model's decision variables, intermediate variables, constants,
and mathematical operations.

See the :ref:`Symbols <opt_model_construction_nl_symbols>` section for an
introduction to working with symbols.

All symbols inherit from the :class:`Symbol` class and therefore inherit its
methods.

Most mathematical symbols inherit from the :class:`ArraySymbol` class and
therefore inherit its methods.

All symbols listed on the :ref:`Model Symbols <symbols_model_symbols>`
page inherit from the :class:`Symbol` class and, for most
mathematical symbols, the :class:`ArraySymbol` class.

.. _symbols_base_symbols:

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    Symbol
    ArraySymbol

