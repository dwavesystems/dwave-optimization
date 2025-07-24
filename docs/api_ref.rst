.. _optimization_api_ref:

=============
API Reference
=============

Nonlinear Models
================

The `dwave-optimization` package provides the
:class:`~dwave.optimization.model.Model` and
:class:`~dwave.optimization.model.States` classes to construct nonlinear models
and handle results, respectively. These models map to a
`directed acyclic graph <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_
constituted of the package's :ref:`optimization_symbols` classes.

For examples, see the :ref:`opt_index_examples_beginner` section.

Models and States
-----------------

.. automodule:: dwave.optimization.model

.. currentmodule:: dwave.optimization

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    ~model.Model
    ~model.States

.. _optimization_generators:

Model Generators
~~~~~~~~~~~~~~~~

.. autosummary::
   :recursive:
   :toctree: generated/
   :template: autosummary_module_functions.rst

   generators

.. _optimization_math:

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :recursive:
   :toctree: generated/
   :template: autosummary_module_functions.rst

   mathematical

.. _optimization_symbols:

Symbols
-------

Symbols are a model's decision variables, intermediate variables, constants,
and mathematical operations.

See the :ref:`Symbols <opt_model_construction_nl_symbols>` section for an
introduction to working with symbols.

All symbols inherit from the :class:`~dwave.optimization.model.Symbol` class and
therefore inherit its methods.

Most mathematical symbols inherit from the
:class:`~dwave.optimization.model.ArraySymbol` class and therefore inherit its
methods.

All symbols listed on the :ref:`Model Symbols <symbols_model_symbols>`
page inherit from the :class:`~dwave.optimization.model.Symbol` class and, for
most mathematical symbols, the :class:`~dwave.optimization.model.ArraySymbol`
class.

.. _symbols_base_symbols:

.. autosummary::
    :recursive:
    :nosignatures:
    :toctree: generated
    :template: autosummary_class.rst

    ~model.Symbol
    ~model.ArraySymbol

.. _symbols_model_symbols:

Model Symbols
~~~~~~~~~~~~~

Each operation, decision, constant, mathematical function, and
flow control is modeled using a symbol. The following symbols
are available for modelling.

In general, symbols should be created using the methods inherited from
:class:`Symbol` and :class:`ArraySymbol`, rather than by the constructors
of the following classes.

.. autosummary::
   :recursive:
   :toctree: generated/
   :template: autosummary_module_classes.rst

    symbols
