# Copyright 2024 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import abc
import itertools
import math
import typing
import unittest

import numpy as np

import dwave.optimization
import dwave.optimization.symbols
from dwave.optimization import Model, logical_or, logical_and


class SymbolTestsMixin(abc.ABC):
    @abc.abstractmethod
    def generate_symbols(self):
        """Yield symbol(s) for testing.

        The model must be topologically sorted before returning.
        The symbols must all be unique from eachother.
        """
    @abc.abstractmethod
    def assertFalse(self, *args, **kwargs): ...

    @abc.abstractmethod
    def assertGreaterEqual(self, *args, **kwargs): ...

    @abc.abstractmethod
    def assertIn(self, *args, **kwargs): ...

    @abc.abstractmethod
    def assertIs(self, *args, **kwargs): ...

    @abc.abstractmethod
    def assertLessEqual(self, *args, **kwargs): ...

    @abc.abstractmethod
    def assertTrue(self, *args, **kwargs): ...

    def test_equality(self):
        for x in self.generate_symbols():
            self.assertTrue(x.maybe_equals(x))
            self.assertTrue(x.equals(x))

        for x, y in zip(self.generate_symbols(), self.generate_symbols()):
            self.assertTrue(x.maybe_equals(y))
            self.assertTrue(x.equals(y))

    def test_inequality(self):
        for x, y in itertools.combinations(self.generate_symbols(), 2):
            self.assertLessEqual(x.maybe_equals(y), 1)
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
        for x in self.generate_symbols():
            model = x.model
            index = x.topological_index()

            with model.to_file() as f:
                new = Model.from_file(f)

            # Get the symbol back
            y, = itertools.islice(new.iter_symbols(), index, index+1)

            self.assertFalse(x.shares_memory(y))
            self.assertIs(type(x), type(y))
            self.assertTrue(x.equals(y))

    def test_state_serialization(self):
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
            
            with model.states.to_file() as f:
                new = Model.from_file(f)

            for i, sym in enumerate(model.iter_symbols()):
                if hasattr(sym, "state"):
                    np.testing.assert_equal(sym.state(), states[i])

    def test_state_size_smoke(self):
        for x in self.generate_symbols():
            self.assertGreaterEqual(x.state_size(), 0)


class UnaryOpTestsMixin(abc.ABC):
    @abc.abstractmethod
    def op(self, x):
        pass

    @abc.abstractmethod
    def assertEqual(self, *args, **kwargs): ...

    def generate_symbols(self):
        model = Model()
        a = model.constant(-5)
        op_a = self.op(a)
        model.lock()
        yield op_a

    def test_scalar_input(self):
        model = Model()
        a = model.constant(-5)
        op_a = self.op(a)
        self.assertEqual(model.num_symbols(), 2)

        # Should be a scalar
        self.assertEqual(op_a.shape(), ())
        self.assertEqual(op_a.size(), 1)

        model.lock()
        model.states.resize(1)
        self.assertEqual(op_a.state(0), self.op(-5))

    def test_1d_input(self):
        model = Model()
        x = model.integer(10, -5, 5)
        op_x = self.op(x)
        model.lock()

        model.states.resize(1)
        data = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        x.set_state(0, data)

        np.testing.assert_equal(op_x.state(0), [self.op(v) for v in data])


class NaryOpTestsMixin(SymbolTestsMixin):
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


class TestAbs(unittest.TestCase, UnaryOpTestsMixin, SymbolTestsMixin):
    def op(self, x):
        return abs(x)


class TestAdd(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        ab = a + b
        ba = b + a
        model.lock()
        yield ab
        yield ba

    def test_scalar_addition(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        x = a + b
        self.assertEqual(model.num_symbols(), 3)

        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(x.shape(), b.shape())
        self.assertEqual(x.size(), a.size())
        self.assertEqual(x.size(), b.size())

        model.lock()
        model.states.resize(1)
        self.assertEqual(x.state(0), 12)

    def test_unlike_shapes(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(ValueError):
            a + b


class TestAll(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        nodes = [
            model.constant(0),
            model.constant(1),
            model.constant([0, 0]),
            model.constant([0, 1]),
            model.constant([1, 1]),
        ]

        model.lock()
        yield from nodes

    def test_scalar(self):
        model = Model()
        model.states.resize(1)

        for val in [0, .0001, 1, 7]:
            with self.subTest(f"[{val}].all()"):
                symbol = model.constant([val]).all()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), bool(val))
                model.unlock()

            with self.subTest(f"({val}).all()"):
                symbol = model.constant(val).all()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), bool(val))
                model.unlock()

        for arr in [np.zeros(5), np.ones(5), np.asarray([0, 1])]:
            with self.subTest(f"[{arr}].all()"):
                symbol = model.constant([arr]).all()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), arr.all())
                model.unlock()

            with self.subTest(f"{val}.all()"):
                symbol = model.constant(arr).all()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), arr.all())
                model.unlock()


class TestAnd(unittest.TestCase):
    def test_scalar_and(self):
        model = Model()
        a = model.constant(1)
        b = model.constant(1)
        c = model.constant(0)
        d = model.constant(0)
        ab = logical_and(a, b)
        ac = logical_and(a, c)
        cd = logical_and(c, d)
        cb = logical_and(c, b)
        self.assertEqual(model.num_symbols(), 8)

        model.lock()
        model.states.resize(1)
        self.assertEqual(ab.state(0), 1)
        self.assertEqual(ac.state(0), 0)
        self.assertEqual(cd.state(0), 0)
        self.assertEqual(cb.state(0), 0)

        with self.assertRaises(TypeError):
            x = a & b
        with self.assertRaises(AttributeError):
            x = a.logical_and(b)


class TestArrayValidation(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        x = model.binary(10)
        a = dwave.optimization.symbols._ArrayValidation(x)
        model.lock()
        yield a

    def test_namespace(self):
        pass


class TestBasicIndexing(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        symbols = []

        model = Model()
        x = model.binary(10)
        y = model.set(10)
        z = model.binary((5, 5))

        symbols.append(x[::2])
        symbols.append(x[0])
        symbols.append(y[:4])
        symbols.append(z[0, 3])
        symbols.append(z[::2, 4])

        model.lock()

        yield from symbols

    def test_infer_indices_1d(self):
        model = Model()
        x = model.binary(10)

        self.assertEqual(x[:]._infer_indices(), (slice(0, 10, 1),))
        self.assertEqual(x[1::]._infer_indices(), (slice(1, 10, 1),))
        self.assertEqual(x[:3:]._infer_indices(), (slice(0, 3, 1),))
        self.assertEqual(x[::2]._infer_indices(), (slice(0, 10, 2),))
        self.assertEqual(x[1::2]._infer_indices(), (slice(1, 10, 2),))
        self.assertEqual(x[2::2]._infer_indices(), (slice(2, 10, 2),))
        self.assertEqual(x[::100]._infer_indices(), (slice(0, 10, 100),))
        self.assertEqual(x[100::100]._infer_indices(), (slice(10, 10, 100),))
        self.assertEqual(x[:3:100]._infer_indices(), (slice(0, 10, 100),))  # harmless

        self.assertEqual(x[0]._infer_indices(), (0,))
        self.assertEqual(x[5]._infer_indices(), (5,))

    def test_infer_indices_2d(self):
        model = Model()
        x = model.binary(shape=(5, 6))

        self.assertEqual(x[3, 4]._infer_indices(), (3, 4))
        self.assertEqual(x[0, 4]._infer_indices(), (0, 4))
        self.assertEqual(x[0, 0]._infer_indices(), (0, 0))

        self.assertEqual(x[3, 4::2]._infer_indices(), (3, slice(4, 6, 2)))
        self.assertEqual(x[3, 4:4:2]._infer_indices(), (3, slice(4, 4, 2)))

        self.assertEqual(x[:, :]._infer_indices(), (slice(0, 5, 1), slice(0, 6, 1)))
        self.assertEqual(x[::2, :]._infer_indices(), (slice(0, 5, 2), slice(0, 6, 1)))
        self.assertEqual(x[:, ::2]._infer_indices(), (slice(0, 5, 1), slice(0, 6, 2)))
        self.assertEqual(x[2:, ::2]._infer_indices(), (slice(2, 5, 1), slice(0, 6, 2)))

    def test_infer_indices_3d(self):
        model = Model()
        x = model.binary(shape=(5, 6, 7))

        self.assertEqual(x[3, 4, 0]._infer_indices(), (3, 4, 0))
        self.assertEqual(x[0, 4, 0]._infer_indices(), (0, 4, 0))
        self.assertEqual(x[0, 0, 0]._infer_indices(), (0, 0, 0))

        self.assertEqual(x[:]._infer_indices(), (slice(0, 5, 1), slice(0, 6, 1), slice(0, 7, 1)))
        self.assertEqual(x[:, 3, :]._infer_indices(), (slice(0, 5, 1), 3, slice(0, 7, 1)))
        self.assertEqual(x[:, :, 3]._infer_indices(), (slice(0, 5, 1), slice(0, 6, 1), 3))

    def test_infer_indices_1d_dynamic(self):
        MAX = np.iinfo(np.intp).max  # constant for unbounded slices

        model = Model()
        x = model.set(10)

        self.assertEqual(x[:]._infer_indices(), (slice(0, MAX, 1),))
        self.assertEqual(x[::2]._infer_indices(), (slice(0, MAX, 2),))
        self.assertEqual(x[5:2:2]._infer_indices(), (slice(5, 2, 2),))
        self.assertEqual(x[:2:]._infer_indices(), (slice(0, 2, 1),))

    def test_state_size(self):
        model = Model()
        self.assertEqual(model.set(10)[::2].state_size(), 5 * 8)


class TestBinaryVariable(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        x = model.binary(10)
        y = model.binary((3, 3))
        model.lock()
        yield x
        yield y

        model = Model()
        z = model.binary([5])
        model.lock()
        yield z

    def test(self):
        model = Model()

        x = model.binary([10])

    def test_no_shape(self):
        model = Model()
        x = model.binary()

        self.assertEqual(x.shape(), tuple())
        model.states.resize(1)
        self.assertEqual(x.state(0).shape, tuple())

    def test_subscript(self):
        model = Model()

        x = model.binary(5)

        self.assertEqual(x[0].shape(), tuple())
        self.assertEqual(x[1:3].shape(), (2,))

    # Todo: we can generalize many of these tests for all decisions that can have
    # their state set

    def test_shape(self):
        model = Model()

        with self.assertRaises(TypeError):
            model.binary(3.5)

        with self.assertRaises(ValueError):
            model.binary([0.5])

    def test_set_state(self):
        with self.subTest("array-like"):
            model = Model()
            model.states.resize(1)
            x = model.binary([5, 5])

            x.set_state(0, np.arange(25) % 2)
            np.testing.assert_array_equal(x.state(), np.arange(25).reshape((5, 5)) % 2)
            x.set_state(0, 1 - np.arange(25).reshape((5, 5)) % 2)
            np.testing.assert_array_equal(x.state(), 1 - np.arange(25).reshape((5, 5)) % 2)

        with self.subTest("invalid state index"):
            model = Model()
            x = model.binary(5)

            # No states have been created
            with self.assertRaisesRegex(ValueError, r"^index out of range: 0$"):
                x.set_state(0, range(5))
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, range(5))

            # Some states have been created
            model.states.resize(5)
            with self.assertRaisesRegex(ValueError, r"^index out of range: 5$"):
                x.set_state(5, range(5))
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, range(5))

        with self.subTest("non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            x = model.binary(5)

            x.set_state(0, [0.5, 0.75, 0.5, 1.0, 0.1])
            np.testing.assert_array_equal(x.state(), [0, 0, 0, 1, 0])

        with self.subTest("invalid"):
            model = Model()
            model.states.resize(1)
            x = model.binary([5])

            # wrong entries
            with self.assertRaisesRegex(ValueError, r"Invalid data provided for node"):
                x.set_state(0, [0, 0, 1, 2, 0])

            # wrong size
            with self.assertRaises(ValueError):
                x.set_state(0, [0, 1, 2])


class TestConstant(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(25, dtype=np.double).reshape((5, 5)))
        B = model.constant([0, 1, 2, 3, 4])
        C = model.constant(np.arange(25, dtype=np.double))
        D = model.constant(np.arange(1, 26, dtype=np.double).reshape((5, 5)))
        model.lock()
        yield A
        yield B

    def test_copy(self):
        model = Model()

        arr = np.arange(25, dtype=np.double).reshape((5, 5))
        A = model.constant(arr)

        np.testing.assert_array_equal(A, arr)
        self.assertTrue(np.shares_memory(A, arr))

    def test_noncontiguous(self):
        model = Model()
        c = model.constant(np.arange(6)[::2])
        np.testing.assert_array_equal(c, [0, 2, 4])

    def test_scalar(self):
        model = Model()
        c = model.constant(5)
        self.assertEqual(c.shape(), tuple())
        np.testing.assert_array_equal(c, 5)


class TestDisjointBitSetsVariable(unittest.TestCase, SymbolTestsMixin):
    def test_inequality(self):
        # TODO re-enable this once equality has been fixed
        pass

    def generate_symbols(self):
        model = Model()
        d, ds = model.disjoint_bit_sets(10, 4)
        model.lock()
        yield d
        yield from ds

    def test(self):
        model = Model()

        model.disjoint_bit_sets(10, 4)

    def test_num_returned_nodes(self):
        model = Model()

        d, ds = model.disjoint_bit_sets(10, 4)

    def test_set_state(self):
        with self.subTest("array-like output lists"):
            model = Model()
            model.states.resize(1)
            x, ys = model.disjoint_bit_sets(5, 3)
            model.lock()

            x.set_state(0, [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]])

            np.testing.assert_array_equal(ys[0].state(), [1, 1, 0, 0, 0])
            np.testing.assert_array_equal(ys[1].state(), [0, 0, 1, 1, 0])
            np.testing.assert_array_equal(ys[2].state(), [0, 0, 0, 0, 1])

        with self.subTest("invalid state index"):
            model = Model()
            x, _ = model.disjoint_bit_sets(5, 3)

            state = [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]]

            # No states have been created
            with self.assertRaisesRegex(ValueError, r"^index out of range: 0$"):
                x.set_state(0, state)
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, state)

            # Some states have been created
            model.states.resize(5)
            with self.assertRaisesRegex(ValueError, r"^index out of range: 5$"):
                x.set_state(5, state)
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, state)

        with self.subTest("non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            x, ys = model.disjoint_bit_sets(5, 3)
            model.lock()

            x.set_state(0, [[1.1, 1, 0.0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]])
            np.testing.assert_array_equal(ys[0].state(), [1, 1, 0, 0, 0])

        with self.subTest("invalid"):
            model = Model()
            model.states.resize(1)
            x, ys = model.disjoint_bit_sets(5, 3)
            model.lock()

            with self.assertRaisesRegex(
                ValueError,
                r"^disjoint set elements must be in exactly one bit-set once$"
            ):
                x.set_state(0, [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

            with self.assertRaisesRegex(
                ValueError,
                r"^disjoint set elements must be in exactly one bit-set once$"
            ):
                x.set_state(0, [[0, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

            # wrong number of lists
            with self.assertRaises(ValueError):
                x.set_state(0, [
                    [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
                ])

    def test_state_serialization_explicit(self):
        model = Model()
        model.states.resize(1)
        x, ys = model.disjoint_bit_sets(5, 3)
        model.lock()

        x.set_state(0, [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]])
        with model.states.to_file() as f:
            model.states.from_file(f)

        np.testing.assert_array_equal(ys[0].state(), [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(ys[1].state(), [0, 0, 1, 1, 0])
        np.testing.assert_array_equal(ys[2].state(), [0, 0, 0, 0, 1])


class TestDisjointListsVariable(unittest.TestCase, SymbolTestsMixin):
    def test_inequality(self):
        # TODO re-enable this once equality has been fixed
        pass

    def generate_symbols(self):
        model = Model()
        d, ds = model.disjoint_lists(10, 4)
        model.lock()
        yield d
        yield from ds

    def test(self):
        model = Model()

        model.disjoint_lists(10, 4)

    def test_num_returned_nodes(self):
        model = Model()

        d, ds = model.disjoint_lists(10, 4)

    def test_set_state(self):
        with self.subTest("array-like output lists"):
            model = Model()
            model.states.resize(1)
            x, ys = model.disjoint_lists(5, 3)
            model.lock()

            x.set_state(0, [[0, 1], [2, 3], [4]])

            np.testing.assert_array_equal(ys[0].state(), [0, 1])
            np.testing.assert_array_equal(ys[1].state(), [2, 3])
            np.testing.assert_array_equal(ys[2].state(), [4])

        with self.subTest("invalid state index"):
            model = Model()
            x, _ = model.disjoint_lists(5, 3)

            state = [[0, 1, 2, 3, 4], [], []]

            # No states have been created
            with self.assertRaisesRegex(ValueError, r"^index out of range: 0$"):
                x.set_state(0, state)
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, state)

            # Some states have been created
            model.states.resize(5)
            with self.assertRaisesRegex(ValueError, r"^index out of range: 5$"):
                x.set_state(5, state)
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, state)

        with self.subTest("non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            x, ys = model.disjoint_lists(5, 3)
            model.lock()

            x.set_state(0, [[4.5, 3, 2, 1, 0], [], []])
            np.testing.assert_array_equal(ys[0].state(), [4, 3, 2, 1, 0])

        with self.subTest("invalid"):
            model = Model()
            model.states.resize(1)
            x, ys = model.disjoint_lists(5, 3)
            model.lock()

            with self.assertRaisesRegex(
                ValueError, r"^disjoint list elements must be in exactly one list once$"
            ):
                x.set_state(0, [[0, 0, 1, 2, 3], [], []])

            with self.assertRaisesRegex(
                ValueError,
                r"^disjoint list elements must be in exactly one list once$"
            ):
                x.set_state(0, [[0, 1, 2, 3], [3], [4]])

            with self.assertRaisesRegex(
                ValueError, (
                    r"^disjoint lists must contain all elements in the range "
                    r"\[0, primary_set_size\)$"
                )
            ):
                x.set_state(0, [[0, 1, 2], [], [4]])

            # wrong number of lists
            with self.assertRaises(ValueError):
                x.set_state(0, [[0, 1, 2, 3, 4], [], [], []])

    def test_state_size(self):
        model = Model()

        d, ds = model.disjoint_lists(10, 4)

        self.assertEqual(d.state_size(), 0)
        for s in ds:
            self.assertEqual(s.state_size(), 10 * 8)


class TestIntegerVariable(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        x = model.integer(10)
        y = model.integer((3, 3))
        model.lock()
        yield x
        yield y

        model = Model()
        z = model.integer([5])
        model.lock()
        yield z

    def test(self):
        model = Model()

        x = model.integer([10])

    def test_subscript(self):
        model = Model()

        x = model.integer(5)

        self.assertEqual(x[0].shape(), tuple())
        self.assertEqual(x[1:3].shape(), (2,))

    def test_no_shape(self):
        model = Model()
        x = model.integer()

        self.assertEqual(x.shape(), tuple())
        model.states.resize(1)
        self.assertEqual(x.state(0).shape, tuple())

    def test_bounds(self):
        model = Model()
        x = model.integer(lower_bound=4, upper_bound=5)
        self.assertEqual(x.lower_bound(), 4)
        self.assertEqual(x.upper_bound(), 5)

    def test_lower_bound(self):
        model = Model()
        x = model.integer(lower_bound=5)
        self.assertEqual(x.lower_bound(), 5)

    def test_upper_bound(self):
        model = Model()
        x = model.integer(upper_bound=5)
        self.assertEqual(x.upper_bound(), 5)

    # Todo: we can generalize many of these tests for all decisions that can have
    # their state set

    def test_shape(self):
        model = Model()

        with self.assertRaises(TypeError):
            model.binary(3.5)

        with self.assertRaises(ValueError):
            model.binary([0.5])

    def test_serialization(self):
        model = Model()
        integers = [
            model.integer((5, 2)),
            model.integer(3, lower_bound=-1),
            model.integer(upper_bound=105),
            model.integer(15, lower_bound=4, upper_bound=6),
        ]

        model.lock()
        with model.to_file() as f:
            copy = Model.from_file(f)

        for old, new in zip(integers, copy.iter_decisions()):
            self.assertEqual(old.shape(), new.shape())
            self.assertEqual(old.lower_bound(), new.lower_bound())
            self.assertEqual(old.upper_bound(), new.upper_bound())

    def test_set_state(self):
        with self.subTest("Simple positive integer"):
            model = Model()
            model.states.resize(1)
            x = model.integer(1)
            x.set_state(0, 1234)
            np.testing.assert_equal(x.state(), 1234)

        with self.subTest("Simple negative integer"):
            model = Model()
            model.states.resize(1)
            x = model.integer(1, lower_bound=-2000, upper_bound=2000)
            x.set_state(0, -1234)
            np.testing.assert_equal(x.state(), -1234)

        with self.subTest("Default bounds test"):
            model = Model()
            model.states.resize(1)
            x = model.integer(1)
            x.set_state(0, 1234)
            with np.testing.assert_raises(ValueError):
                x.set_state(0, -1234)

        with self.subTest("Simple bounds test"):
            model = Model()
            model.states.resize(1)
            x = model.integer(1, lower_bound=-1, upper_bound=1)
            x.set_state(0, 1)
            x.set_state(0, -1)
            with np.testing.assert_raises(ValueError):
                x.set_state(0, 2)
            with np.testing.assert_raises(ValueError):
                x.set_state(0, -2)

        with self.subTest("array-like"):
            model = Model()
            model.states.resize(1)
            x = model.integer([5, 5], lower_bound=-100, upper_bound=100)

            x.set_state(0, np.arange(25))
            np.testing.assert_array_equal(x.state(), np.arange(25).reshape((5, 5)))
            x.set_state(0, 1 - np.arange(25).reshape((5, 5)))
            np.testing.assert_array_equal(x.state(), 1 - np.arange(25).reshape((5, 5)))

        with self.subTest("positive non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            x = model.integer(5)

            x.set_state(0, [0.5, 0.75, 0.5, 1.0, 0.1])
            np.testing.assert_array_equal(x.state(), [0, 0, 0, 1, 0])

        with self.subTest("negative non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            x = model.integer(5, lower_bound=-1, upper_bound=0)

            x.set_state(0, [-0.5, -0.75, -0.5, -1.0, -0.1])
            np.testing.assert_array_equal(x.state(), [0, 0, 0, -1, 0])


class TestLessEqual(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        a_le_b = a <= b
        b_le_a = b <= a
        model.lock()
        yield a_le_b
        yield b_le_a

    def test_scalar_le(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        x = a <= b
        self.assertEqual(model.num_symbols(), 3)

        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(x.shape(), b.shape())
        self.assertEqual(x.size(), a.size())
        self.assertEqual(x.size(), b.size())

        model.lock()
        model.states.resize(1)
        self.assertEqual(x.state(0), 1)

    def test_1d_le(self):
        model = Model()
        a = model.constant([0, 5, 10])
        b = model.constant([3, 1, 10])
        x = a <= b
        self.assertEqual(model.num_symbols(), 3)
        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(x.shape(), b.shape())
        self.assertEqual(x.size(), a.size())
        self.assertEqual(x.size(), b.size())

        model.lock()
        model.states.resize(1)
        np.testing.assert_array_equal(x.state(), (1, 0, 1))

    def test_unlike_shapes(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(ValueError):
            a <= b


class TestListVariable(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        x = model.list(10)
        y = model.list(0)
        model.lock()
        yield x
        yield y

        model = Model()
        z = model.list(5)
        model.lock()
        yield z

    def test(self):
        model = Model()

        x = model.list(10)

    def test_subscript(self):
        model = Model()

        x = model.list(5)

        self.assertEqual(x[0].shape(), tuple())
        self.assertEqual(x[1:3].shape(), (2,))

    # Todo: we can generalize many of these tests for all decisions that can have
    # their state set

    def test_set_state(self):
        with self.subTest("array-like"):
            model = Model()
            model.states.resize(1)
            x = model.list(5)

            x.set_state(0, range(5))
            np.testing.assert_array_equal(x.state(), range(5))
            x.set_state(0, [4, 3, 2, 1, 0])
            np.testing.assert_array_equal(x.state(), [4, 3, 2, 1, 0])
            x.set_state(0, (2, 1, 0, 3, 4))
            np.testing.assert_array_equal(x.state(), (2, 1, 0, 3, 4))

        with self.subTest("invalid state index"):
            model = Model()
            x = model.list(5)

            # No states have been created
            with self.assertRaisesRegex(ValueError, r"^index out of range: 0$"):
                x.set_state(0, range(5))
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, range(5))

            # Some states have been created
            model.states.resize(5)
            with self.assertRaisesRegex(ValueError, r"^index out of range: 5$"):
                x.set_state(5, range(5))
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                x.set_state(-1, range(5))

        with self.subTest("non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            x = model.list(5)

            x.set_state(0, [4.5, 3, 2, 1, 0])
            np.testing.assert_array_equal(x.state(), [4, 3, 2, 1, 0])

        with self.subTest("invalid"):
            model = Model()
            model.states.resize(1)
            x = model.list(5)

            # repeated entries
            with self.assertRaisesRegex(ValueError, r"^contents must be unique$"):
                x.set_state(0, [0, 0, 1, 2, 3])

            # wrong size
            with self.assertRaises(ValueError):
                x.set_state(0, [0, 1, 2])


class TestMax(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.max()
        b = B.max()
        model.lock()
        yield a
        yield b

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.max()
        b = B.max()
        model.states.resize(1)
        model.lock()

        self.assertEqual(a.state(0), 4)
        self.assertEqual(b.state(0), 9)


class TestMaximum(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        m1 = dwave.optimization.maximum(A, B)
        m2 = dwave.optimization.maximum(B, A)
        model.lock()
        yield m1
        yield m2

    def test(self):
        from dwave.optimization.symbols import Maximum
        model = Model()

        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        m = dwave.optimization.maximum(A, B)

        self.assertIsInstance(m, Maximum)

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(5, 10))
        x = model.list(5)
        m = dwave.optimization.maximum(A, x)
        model.states.resize(1)
        model.lock()

        np.testing.assert_array_equal(m.state(0), np.arange(5, 10))


class TestMin(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.min()
        b = B.min()
        model.lock()
        yield a
        yield b

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.min()
        b = B.min()
        model.states.resize(1)
        model.lock()

        self.assertEqual(a.state(0), 0)
        self.assertEqual(b.state(0), 5)


class TestMinimum(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        m1 = dwave.optimization.minimum(A, B)
        m2 = dwave.optimization.minimum(B, A)
        model.lock()
        yield m1
        yield m2

    def test(self):
        from dwave.optimization.symbols import Minimum
        model = Model()

        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        m = dwave.optimization.minimum(A, B)

        self.assertIsInstance(m, Minimum)

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(5, 10))
        x = model.list(5)
        m = dwave.optimization.minimum(A, x)
        model.states.resize(1)
        model.lock()

        np.testing.assert_array_equal(m.state(0), x.state(0))


class TestMultiply(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        ab = a * b
        ba = b * a
        model.lock()
        yield ab
        yield ba

    def test_scalar_addition(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        x = a * b
        self.assertEqual(model.num_symbols(), 3)

        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(x.shape(), b.shape())
        self.assertEqual(x.size(), a.size())
        self.assertEqual(x.size(), b.size())

        model.lock()
        model.states.resize(1)
        self.assertEqual(x.state(0), 35)

    def test_unlike_shapes(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(ValueError):
            a * b


class TestNaryAdd(unittest.TestCase, NaryOpTestsMixin):
    def op(self, *xs):
        return sum(xs)

    def node_op(self, *xs):
        return dwave.optimization.add(*xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryAdd


class TestNaryMaximum(unittest.TestCase, NaryOpTestsMixin):
    def op(self, *xs):
        return max(xs)

    def node_op(self, *xs):
        return dwave.optimization.maximum(xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryMaximum


class TestNaryMinimum(unittest.TestCase, NaryOpTestsMixin):
    def op(self, *xs):
        return min(xs)

    def node_op(self, *xs):
        return dwave.optimization.minimum(xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryMinimum


class TestNaryMultiply(unittest.TestCase, NaryOpTestsMixin):
    def op(self, *xs):
        return math.prod(xs)

    def node_op(self, *xs):
        return dwave.optimization.multiply(xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryMultiply


class TestNegate(unittest.TestCase, UnaryOpTestsMixin, SymbolTestsMixin):
    def op(self, x):
        return -x


class TestOr(unittest.TestCase):
    def test_scalar_or(self):
        model = Model()
        a = model.constant(1)
        b = model.constant(1)
        c = model.constant(0)
        d = model.constant(0)
        ab = logical_or(a, b)
        ac = logical_or(a, c)
        cd = logical_or(c, d)
        cb = logical_or(c, b)
        self.assertEqual(model.num_symbols(), 8)

        model.lock()
        model.states.resize(1)
        self.assertEqual(ab.state(0), 1)
        self.assertEqual(ac.state(0), 1)
        self.assertEqual(cd.state(0), 0)
        self.assertEqual(cb.state(0), 1)

        with self.assertRaises(TypeError):
            x = a | b
        with self.assertRaises(AttributeError):
            x = a.logical_or(b)

    def test_array_or(self):
        model = Model()
        a = model.constant([ 1, 0, 1, 0 ])
        b = model.constant([ 1, 1, 1, 1 ])
        c = model.constant([ 0, 0, 0, 0 ])
        ab = logical_or(a, b)
        ac = logical_or(a, c)

        model.lock()
        model.states.resize(1)

        np.testing.assert_array_equal(ab.state(0), [ 1, 1, 1, 1 ])
        np.testing.assert_array_equal(ac.state(0), [ 1, 0, 1, 0 ])


class TestPermutation(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(25).reshape((5, 5)))
        x = model.list(5)
        p = A[x, :][:, x]
        model.lock()
        yield p

    def test(self):
        from dwave.optimization.symbols import Permutation

        model = Model()

        A = model.constant(np.arange(25).reshape((5, 5)))
        x = model.list(5)

        self.assertIsInstance(A[x, :][:, x], Permutation)
        self.assertIsInstance(A[:, x][x, :], Permutation)

        b = model.constant(np.arange(30).reshape((5, 6)))

        self.assertNotIsInstance(b[x, :][:, x], Permutation)
        self.assertNotIsInstance(b[:, x][x, :], Permutation)


class TestProd(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.prod()
        b = B.prod()
        model.lock()
        yield a
        yield b

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.prod()
        b = B.prod()
        model.states.resize(1)
        model.lock()

        self.assertEqual(a.state(0), 0)
        self.assertEqual(b.state(0), 5*6*7*8*9)


class TestQuadraticModel(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        num_variables = 5
        model = Model()
        x = model.binary(num_variables)
        y = model.integer(num_variables)
        Q = dict()

        for i in range(num_variables):
            Q[i, i] = i
            for j in range(i+1, num_variables):
                Q[i, j] = i+j

        xQx = model.quadratic_model(x, Q)
        yQy = model.quadratic_model(y, Q)
        model.lock()

        yield xQx
        yield yQy

    def test_exceptions(self):
        model = Model()

        x = model.binary(1)

        with self.assertRaises(ValueError):
            model.quadratic_model(x, {})

        with self.assertRaises(ValueError):
            model.quadratic_model(x, tuple())

        with self.assertRaises(ValueError):
            model.quadratic_model(x, ([[]], []))

        with self.assertRaises(ValueError):
            model.quadratic_model(x, ([], []))

    def test_qubo(self):
        num_variables = 5
        model = Model()
        x = model.binary(num_variables)

        Q = dict()
        for i in range(num_variables):
            Q[i, i] = i
            for j in range(i+1, num_variables):
                Q[i, j] = i+j

        q = model.quadratic_model(x, Q)
        q_with_linear = model.quadratic_model(x, Q, {v: -2 * v for v in range(num_variables)})
        model.lock()

        with self.assertRaises(ValueError):
            q.get_linear(-1)
        with self.assertRaises(ValueError):
            q.get_linear(5)

        for v in range(num_variables):
            self.assertEqual(q.get_linear(v), v)
            self.assertEqual(q_with_linear.get_linear(v), -v)
        for u, v in itertools.combinations(range(num_variables), 2):
            self.assertEqual(q.get_quadratic(u, v), u + v)
            self.assertEqual(q_with_linear.get_quadratic(u, v), u + v)

        self.assertEqual(model.num_nodes(), 3)

    def test_iqp(self):
        num_variables = 5
        model = Model()
        x = model.integer(num_variables, 0, 10)

        Q = dict()
        for i in range(num_variables):
            Q[i, i] = i
            for j in range(i+1, num_variables):
                Q[i, j] = i+j

        q = model.quadratic_model(x, Q)
        model.lock()

        for v in range(num_variables):
            self.assertEqual(q.get_linear(v), 0)
            self.assertEqual(q.get_quadratic(v, v), v)
        for u, v in itertools.combinations(range(num_variables), 2):
            self.assertEqual(q.get_quadratic(u, v), u + v)

        self.assertEqual(model.num_nodes(), 2)


class TestReshape(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(12))
        syms = [A.reshape(12), A.reshape((2, 6)), A.reshape(3, 4)]
        model.lock()
        yield from syms


class TestSquare(unittest.TestCase, UnaryOpTestsMixin, SymbolTestsMixin):
    def op(self, x):
        return x ** 2


class TestSetVariable(unittest.TestCase, SymbolTestsMixin):
    def generate_symbols(self) -> typing.Iterator[dwave.optimization.symbols.SetVariable]:
        # Typical
        model = Model()
        s = model.set(10)
        model.lock()
        yield s

        # Empty
        model = Model()
        s = model.set(0)
        model.lock()
        yield s

        # Fixed-size
        model = Model()
        s = model.set(5, 5)
        model.lock()
        yield s

    def test_shape(self):
        model = Model()

        s = model.set(10)
        self.assertEqual(s.shape(), (-1,))
        self.assertEqual(s.size(), -1)
        self.assertEqual(s.strides(), (np.dtype(np.double).itemsize,))

        t = model.set(5, 5)  # this is exactly range(5)
        self.assertEqual(t.shape(), (5,))
        self.assertEqual(t.size(), 5)
        self.assertEqual(t.strides(), (np.dtype(np.double).itemsize,))

    def test_set_state(self):
        with self.subTest("array-like"):
            model = Model()
            model.states.resize(1)
            s = model.set(5)

            s.set_state(0, range(5))
            np.testing.assert_array_equal(s.state(), range(5))
            s.set_state(0, [4, 3, 2, 1, 0])
            np.testing.assert_array_equal(s.state(), [4, 3, 2, 1, 0])
            s.set_state(0, (2, 1, 0, 3, 4))
            np.testing.assert_array_equal(s.state(), (2, 1, 0, 3, 4))

        with self.subTest("set"):
            model = Model()
            model.states.resize(1)
            s = model.set(5)

            s.set_state(0, set(range(5)))
            np.testing.assert_array_equal(s.state(), range(5))
            s.set_state(0, set([4, 3, 2, 1, 0]))
            np.testing.assert_array_equal(s.state(), range(5))

        with self.subTest("invalid state index"):
            model = Model()
            s = model.set(5)

            # No states have been created
            with self.assertRaisesRegex(ValueError, r"^index out of range: 0$"):
                s.set_state(0, range(5))
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                s.set_state(-1, range(5))

            # Some states have been created
            model.states.resize(5)
            with self.assertRaisesRegex(ValueError, r"^index out of range: 5$"):
                s.set_state(5, range(5))
            with self.assertRaisesRegex(ValueError, r"^index out of range: -1$"):
                s.set_state(-1, range(5))

        with self.subTest("non-integer"):
            # gets translated into integer according to NumPy rules
            model = Model()
            model.states.resize(1)
            s = model.set(5)

            s.set_state(0, [4.5, 3, 2, 1, 0])
            np.testing.assert_array_equal(s.state(), [4, 3, 2, 1, 0])

        with self.subTest("invalid"):
            model = Model()
            model.states.resize(1)
            s = model.set(5)

            # repeated entries
            with self.assertRaisesRegex(ValueError, r"^contents must be unique$"):
                s.set_state(0, [0, 0, 1, 2, 3])

    def test_state_size(self):
        model = Model()
        self.assertEqual(model.set(10).state_size(), 10 * 8)
        self.assertEqual(model.set(10, min_size=5).state_size(), 10 * 8)
        self.assertEqual(model.set(10, max_size=5).state_size(), 5 * 8)


class TestSubtract(unittest.TestCase):
    def test_scalar_subtract(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        x = a - b
        self.assertEqual(model.num_symbols(), 3)

        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(x.shape(), b.shape())
        self.assertEqual(x.size(), a.size())
        self.assertEqual(x.size(), b.size())

        model.lock()
        model.states.resize(1)
        self.assertEqual(x.state(0), -2)

    def test_unlike_shapes(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(ValueError):
            a - b
