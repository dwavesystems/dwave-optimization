.. _optimization_symbols:

=======
Symbols
=======

.. currentmodule:: dwave.optimization.model

Symbols are a model's decision variables, intermediate variables, constants, 
and mathematical operations.

See :ref:`Symbols <intro_optimization_symbols>` for an introduction to
working with symbols.

All symbols listed in the :ref:`Model Symbols<symbols_model_symbols>`
section below inherit from the :class:`Symbol` class and, for most mathematical
symbols, the :class:`ArraySymbol` class.

.. _symbols_base_symbols:

Symbol
======

All symbols inherit from the :class:`Symbol` class and therefore inherit its
methods.

.. autoclass:: Symbol
    :members:
    :member-order: bysource

ArraySymbol
===========

Most mathematical symbols inherit from the :class:`ArraySymbol` class and
therefore inherit its methods.

.. autoclass:: ArraySymbol
    :members:
    :show-inheritance:
    :member-order: bysource

.. _symbols_model_symbols:

Model Symbols
=============

Each operation, decision, constant, mathematical function, and
flow control is modeled using a symbol. The following symbols
are available for modelling.

In general, symbols should be created using the methods inherited from
:class:`Symbol` and :class:`ArraySymbol`, rather than by the constructors
of the following classes.

.. automodule:: dwave.optimization.symbols
    :members:
    :show-inheritance:
