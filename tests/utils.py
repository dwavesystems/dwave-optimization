import abc
import itertools
import unittest

import numpy as np

import dwave.optimization
from dwave.optimization import (
    Model,
)


class SymbolTests(abc.ABC, unittest.TestCase):
    @abc.abstractmethod
    def generate_symbols(self):
        """Yield symbol(s) for testing.

        The model must be topologically sorted before returning.
        The symbols must all be unique from eachother.
        """

    def test_equality(self):
        DEFINITELY = 2
        for x in self.generate_symbols():
            self.assertEqual(DEFINITELY, x.maybe_equals(x))
            self.assertTrue(x.equals(x))

        for x, y in zip(self.generate_symbols(), self.generate_symbols()):
            self.assertTrue(DEFINITELY, x.maybe_equals(y))
            self.assertTrue(x.equals(y))

    def test_inequality(self):
        MAYBE = 1
        for x, y in itertools.combinations(self.generate_symbols(), 2):
            self.assertLessEqual(x.maybe_equals(y), MAYBE)
            self.assertFalse(x.equals(y))

    def test_iter_symbols(self):
        for x in self.generate_symbols():
            model = x.model
            index = x.topological_index()

            # Get the symbol back
            y, = itertools.islice(model.iter_symbols(), index, index+1)

            self.assertTrue(x.shares_memory(y))
            self.assertIs(type(x), type(y))
            self.assertTrue(x.equals(y))

    def test_namespace(self):
        x = next(self.generate_symbols())
        self.assertIn(type(x).__name__, dwave.optimization.symbols.__all__)

    def test_serialization(self):
        for version in dwave.optimization._model.KNOWN_SERIALIZATION_VERSIONS:
            with self.subTest(version=version):
                for x in self.generate_symbols():
                    model = x.model
                    index = x.topological_index()

                    with model.to_file(version=version) as f:
                        new = Model.from_file(f)

                    # Get the symbol back
                    y, = itertools.islice(new.iter_symbols(), index, index+1)

                    self.assertFalse(x.shares_memory(y))
                    self.assertIs(type(x), type(y))
                    self.assertTrue(x.equals(y))

    def test_state_serialization(self):
        for version in dwave.optimization._model.KNOWN_SERIALIZATION_VERSIONS:
            with self.subTest(version=version):
                for x in self.generate_symbols():
                    model = x.model

                    # without some way to randomly initialize states this is really just
                    # smoke test
                    model.states.resize(1)

                    # get the states before serialization
                    states = []
                    for sym in model.iter_symbols():
                        if hasattr(sym, "state"):
                            states.append(sym.state())
                        else:
                            states.append(None)

                    with model.states.to_file(version=version) as f:
                        model.states.clear()
                        model.states.from_file(f)

                    for i, sym in enumerate(model.iter_symbols()):
                        if hasattr(sym, "state"):
                            np.testing.assert_equal(sym.state(), states[i])

    def test_state_size_smoke(self):
        for x in self.generate_symbols():
            self.assertGreaterEqual(x.state_size(), 0)


class BinaryOpTests(SymbolTests):
    @abc.abstractmethod
    def op(self, lhs, rhs):
        pass

    def symbol_op(self, lhs, rhs):
        # if the op is different for symbols, allow the override
        return self.op(lhs, rhs)

    def test_deterministic(self):
        x = next(self.generate_symbols())
        self.assertTrue(x._deterministic_state())

    def test_numpy_equivalence(self):
        lhs_array = np.arange(10)
        rhs_array = np.arange(1, 11)

        model = Model()
        lhs_symbol = model.constant(lhs_array)
        rhs_symbol = model.constant(rhs_array)

        op_array = self.op(lhs_array, rhs_array)
        op_symbol = self.symbol_op(lhs_symbol, rhs_symbol)

        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(op_array, op_symbol.state())

    def test_scalar_broadcasting(self):
        lhs_array = 5
        rhs_array = np.asarray([-10, 100, 16])

        model = Model()
        lhs_symbol = model.constant(lhs_array)
        rhs_symbol = model.constant(rhs_array)

        op_array = self.op(lhs_array, rhs_array)
        op_symbol = self.symbol_op(lhs_symbol, rhs_symbol)

        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(op_array, op_symbol.state())

    def test_size1_broadcasting(self):
        lhs_array = np.asarray([5])
        rhs_array = np.asarray([-10, 100, 16])

        model = Model()
        lhs_symbol = model.constant(lhs_array)
        rhs_symbol = model.constant(rhs_array)

        op_array = self.op(lhs_array, rhs_array)
        op_symbol = self.symbol_op(lhs_symbol, rhs_symbol)

        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(op_array, op_symbol.state())


class NaryOpTests(SymbolTests):
    @abc.abstractmethod
    def op(self, *xs):
        pass

    @abc.abstractmethod
    def node_op(self, *xs):
        pass

    @abc.abstractmethod
    def node_class(self):
        pass

    def generate_symbols(self):
        model = Model()
        a, b, c = model.constant(-5), model.constant(7), model.constant(0)
        op_abc = self.node_op(a, b, c)
        model.lock()
        yield op_abc

    def test_scalar_input(self):
        model = Model()
        a, b, c = model.constant(-5), model.constant(7), model.constant(0)
        op_abc = self.node_op(a, b, c)
        self.assertEqual(model.num_symbols(), 4)

        # Make sure we got the right type of node
        self.assertIsInstance(op_abc, self.node_class())

        # Should be a scalar
        self.assertEqual(op_abc.shape(), ())
        self.assertEqual(op_abc.size(), 1)

        model.lock()
        model.states.resize(1)
        self.assertEqual(op_abc.state(0), self.op(-5, 7, 0))

    def test_1d_input(self):
        model = Model()
        x, y, z = [model.integer(10, -5, 5) for _ in range(3)]
        op_xyz = self.node_op(x, y, z)

        # Make sure we got the right type of node
        self.assertIsInstance(op_xyz, self.node_class())

        # Make sure the shape is correct
        self.assertEqual(op_xyz.shape(), (10,))
        self.assertEqual(op_xyz.size(), 10)

        model.lock()

        model.states.resize(1)

        data_x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        x.set_state(0, data_x)
        data_y = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4]
        y.set_state(0, data_y)
        data_z = [5, -5, 4, -4, 3, -3, 2, -2, 1, 0]
        z.set_state(0, data_z)

        np.testing.assert_equal(
            op_xyz.state(0),
            [self.op(*vs) for vs in zip(data_x, data_y, data_z)]
        )


class UnaryOpTests(SymbolTests):
    @abc.abstractmethod
    def op(self, x):
        pass

    def symbol_op(self, x):
        # if the op is different for symbols, allow the override
        return self.op(x)

    def generate_symbols(self):
        model = Model()
        a = model.constant(-5)
        op_a = self.symbol_op(a)
        model.lock()
        yield op_a

    def test_scalar_input(self):
        for scalar in [-5, -.5, 0, 1, 1.5]:
            with self.subTest(f"a = {scalar}"):
                model = Model()
                a = model.constant(scalar)
                op_a = self.symbol_op(a)
                self.assertEqual(model.num_symbols(), 2)

                # Should be a scalar
                self.assertEqual(op_a.shape(), ())
                self.assertEqual(op_a.size(), 1)

                model.lock()
                model.states.resize(1)
                self.assertEqual(op_a.state(0), self.op(scalar))

    def test_1d_input(self):
        model = Model()
        x = model.integer(10, -5, 5)
        op_x = self.symbol_op(x)
        model.lock()

        model.states.resize(1)
        data = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        x.set_state(0, data)

        np.testing.assert_equal(op_x.state(0), [self.op(v) for v in data])
