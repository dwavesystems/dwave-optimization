.. _optimization_symbols:

=======
Symbols
=======

Symbols are a model's decision variables, intermediate variables, constants, 
and mathematical operations.

See :ref:`Symbols <intro_optimization_symbols>` for an introduction to
working with symbols.

Base Classes
============

.. currentmodule:: dwave.optimization.model

.. _symbols_base_symbols:

.. autoclass:: Symbol

The following :class:`~dwave.optimization.model.Symbol` methods
are inherited by the :class:`~dwave.optimization.model.ArraySymbol`
class and :ref:`model symbols <symbols_model_symbols>`.

.. autosummary::
    :toctree: generated/

    ~Symbol.equals
    ~Symbol.has_state
    ~Symbol.id
    ~Symbol.iter_predecessors
    ~Symbol.iter_successors
    ~Symbol.maybe_equals
    ~Symbol.reset_state
    ~Symbol.shares_memory
    ~Symbol.state_size
    ~Symbol.topological_index

.. autoclass:: ArraySymbol

The following :class:`~dwave.optimization.model.ArraySymbol` methods
are inherited by the :ref:`model symbols <symbols_model_symbols>`.

.. autosummary::
    :toctree: generated/

    ~ArraySymbol.all
    ~ArraySymbol.any
    ~ArraySymbol.has_state
    ~ArraySymbol.max
    ~ArraySymbol.maybe_equals
    ~ArraySymbol.min
    ~ArraySymbol.ndim
    ~ArraySymbol.prod
    ~ArraySymbol.reshape
    ~ArraySymbol.sum
    ~ArraySymbol.shape
    ~ArraySymbol.size
    ~ArraySymbol.state
    ~ArraySymbol.state_size
    ~ArraySymbol.strides

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
