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

import collections.abc
import itertools
import math
import operator
import typing
import unittest

import numpy as np

import dwave.optimization
import dwave.optimization.symbols
from dwave.optimization import (
    Model,
    arange,
    expit,
    log,
    logical,
    logical_and,
    logical_or,
    logical_not,
    logical_xor,
    mod,
    put,
    rint,
    sqrt,
    stack,
)

# Try to import utils normally. If it fails, assume that we are running the
# tests from the inside the `tests/` directory to avoid importing
# dwave.optimization locally.
try:
    import tests.utils as utils
except ImportError:
    import utils  # type: ignore


class TestAbsolute(utils.UnaryOpTests):
    def op(self, x):
        return abs(x)

    def test_abs(self):
        from dwave.optimization.symbols import Absolute

        model = Model()
        x = model.integer(5, lower_bound=-5, upper_bound=5)
        a = abs(x)
        self.assertIsInstance(a, Absolute)
        self.assertEqual(model.num_symbols(), 2)


class TestAdd(utils.BinaryOpTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        ab = a + b
        ba = b + a
        model.lock()
        yield ab
        yield ba

    def op(self, lhs, rhs):
        return lhs + rhs

    def test_broadcasting(self):
        # todo: allow array broadcasting, for now just test that it raises
        # an error
        model = Model()
        a = model.integer(5)
        b = model.integer((5, 5))
        with self.assertRaises(ValueError):
            a + b

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

    def test_scalar_broadcasting(self):
        # todo: allow array broadcasting, for now just test that it raises
        # an error
        model = Model()
        a = model.integer(1)
        b = model.integer(5)
        x = a + b

        model.lock()
        model.states.resize(1)

        a.set_state(0, 3)
        b.set_state(0, [4, 0, 3, 100, 17])
        np.testing.assert_array_equal(x.state(), [7, 3, 6, 103, 20])

    def test_unlike_shapes(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(ValueError):
            a + b


class TestAdvancedIndexing(unittest.TestCase):
    def test_out_of_bounds(self):
        model = Model()

        a = model.constant([0, 1, 2, 3])
        b = model.constant(1.5)
        x = model.integer(lower_bound=1, upper_bound=100)

        # Error messages are chosen to be similar to NumPy's
        with self.assertRaisesRegex(
                IndexError,
                "index's smallest possible value -100 is out of bounds for axis 0 with size 4"):
            a[-x]
        with self.assertRaisesRegex(
                IndexError,
                "index's largest possible value 100 is out of bounds for axis 0 with size 4"):
            a[x]

        with self.assertRaisesRegex(
                IndexError,
                "index may not contain non-integer values for axis 0"):
            a[b]

        a = model.constant(np.arange(12).reshape(3, 4))

        with self.assertRaisesRegex(
                IndexError,
                "index may not contain non-integer values for axis 1"):
            a[:, b]

        s = model.set(5, min_size=3)

        with self.assertRaisesRegex(
                IndexError,
                "index's largest possible value 100 is out of bounds for axis 0 "
                "with minimum size 3"):
            s[x]

    def test_higher_dimenional_indexers_not_allowed(self):
        model = Model()
        constant = model.constant(np.arange(10))

        i0 = model.integer(lower_bound=0, upper_bound=9)
        self.assertTrue(constant[i0].shape() == tuple())

        i1 = model.integer(3, lower_bound=0, upper_bound=9)
        self.assertTrue(constant[i1].shape() == (3,))

        i2 = model.integer((2, 3), lower_bound=0, upper_bound=9)
        with self.assertRaises(ValueError):
            constant[i2]


class TestAll(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        nodes = [
            model.constant(0).all(),
            model.constant(1).all(),
            model.constant([]).all(),
            model.constant([0, 0]).all(),
            model.constant([0, 1]).all(),
            model.constant([1, 1]).all(),
        ]

        model.lock()
        yield from nodes

    def test_empty(self):
        model = Model()
        empty = model.constant([]).all()
        model.lock()
        model.states.resize(1)

        self.assertTrue(empty.state())
        self.assertEqual(empty.state(), np.asarray([]).all())  # confirm consistency with NumPy

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


class TestAnd(utils.BinaryOpTests):
    def generate_symbols(self):
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

        yield from (ab, ac, cd, cb)

    def op(self, lhs, rhs):
        return np.logical_and(lhs, rhs)

    def symbol_op(self, lhs, rhs):
        return logical_and(lhs, rhs)

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
            a & b
        with self.assertRaises(AttributeError):
            a.logical_and(b)


class TestAny(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        nodes = [
            model.constant(0).any(),
            model.constant(1).any(),
            model.constant([]).any(),
            model.constant([0, 0]).any(),
            model.constant([0, 1]).any(),
            model.constant([1, 1]).any(),
        ]

        model.lock()
        yield from nodes

    def test_empty(self):
        model = Model()
        empty = model.constant([]).any()
        model.lock()
        model.states.resize(1)

        self.assertFalse(empty.state())
        self.assertEqual(empty.state(), np.asarray([]).any())  # confirm consistency with NumPy

    def test_scalar(self):
        model = Model()
        model.states.resize(1)

        for val in [0, .0001, 1, 7]:
            with self.subTest(f"[{val}].all()"):
                symbol = model.constant([val]).any()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), bool(val))
                model.unlock()

            with self.subTest(f"({val}).all()"):
                symbol = model.constant(val).any()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), bool(val))
                model.unlock()

        for arr in [np.zeros(5), np.ones(5), np.asarray([0, 1])]:
            with self.subTest(f"[{arr}].all()"):
                symbol = model.constant([arr]).any()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), arr.any())
                model.unlock()

            with self.subTest(f"{val}.all()"):
                symbol = model.constant(arr).any()
                model.lock()
                np.testing.assert_array_equal(symbol.state(0), arr.any())
                model.unlock()


class TestArrayValidation(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        x = model.binary(10)
        a = dwave.optimization.symbols._ArrayValidation(x)
        model.lock()
        yield a

    def test_namespace(self):
        pass


class TestARange(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        nodes = [
            arange(0, model.integer(lower_bound=0, upper_bound=5)),
            # arange(model.integer())
        ]

        with model.lock():
            yield from nodes

    def test_construction_three_arg(self):
        model = Model()
        model.states.resize(1)

        # there are lots of combos for construction to test
        for start in [1, model.constant(1)]:
            for stop in [5, model.constant(5)]:
                for step in [2, model.constant(2)]:
                    if all(isinstance(v, int) for v in (start, stop, step)):
                        with self.assertRaises(ValueError):
                            arange(start, stop, step)
                        continue
                    ar = arange(start, stop, step)
                    with model.lock():
                        np.testing.assert_array_equal(ar.state(), np.arange(1, 5, 2))

                    ar = arange(start, step=step, stop=stop)
                    with model.lock():
                        np.testing.assert_array_equal(ar.state(), np.arange(1, 5, 2))

    def test_construction_two_arg(self):
        model = Model()
        model.states.resize(1)

        # there are lots of combos for construction to test
        for start in [1, model.constant(1)]:
            for stop in [5, model.constant(5)]:
                if all(isinstance(v, int) for v in (start, stop)):
                    with self.assertRaises(ValueError):
                        arange(start, stop)
                    continue
                ar = arange(start, stop)
                with model.lock():
                    np.testing.assert_array_equal(ar.state(), np.arange(1, 5, 1))

                ar = arange(stop=stop, start=start)
                with model.lock():
                    np.testing.assert_array_equal(ar.state(), np.arange(1, 5, 1))


class TestBasicIndexing(utils.SymbolTests):
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


class TestBinaryVariable(utils.SymbolTests):
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

        model.binary([10])

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


class TestConcatenate(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        x = model.constant(np.arange(12)).reshape((2, 1, 2, 3))
        y = model.constant(np.arange(24)).reshape((2, 2, 2, 3))
        z = model.constant(np.arange(36)).reshape((2, 3, 2, 3))
        c = dwave.optimization.symbols.Concatenate((x, y, z), axis=1)
        model.lock()
        yield c

    def test_simple_concatenate(self):
        model = Model()
        with self.subTest("Concatenate ndarray of binary returns Concatenate"):
            A = [model.binary(5), model.binary(5)]
            self.assertIsInstance(
                dwave.optimization.concatenate(tuple(A)),
                dwave.optimization.symbols.Concatenate
            )
        with self.subTest("Concatenate Iterable and Sized of length 1 returns ArraySymbol"):
            self.assertIsInstance(
                dwave.optimization.concatenate((model.binary(5),)),
                dwave.optimization.model.ArraySymbol
            )

    def test_errors(self):
        model = Model()
        with self.subTest("zero dimensions"):
            a = model.constant(1)
            with self.assertRaisesRegex(
                ValueError,
                "axis 0 is out of bounds for array of dimension 0",
            ):
                dwave.optimization.concatenate((a,))

        with self.subTest("same number of dimensions"):
            A = model.constant(np.arange(6)).reshape((1, 2, 3))
            B = model.constant(np.arange(24)).reshape((1, 2, 3, 4))
            with self.assertRaisesRegex(
                ValueError, (r"^all the input arrays must have the same "
                             r"number of dimensions, but the array at index 0 "
                             r"has 3 dimension\(s\) and the array at index 1 "
                             r"has 4 dimension\(s\)")
            ):
                dwave.optimization.symbols.Concatenate((A, B))

        with self.subTest("array shapes are the same"):
            A = model.constant(np.arange(6)).reshape((1, 2, 3))
            B = model.constant(np.arange(6)).reshape((3, 2, 1))
            axis = 1
            with self.assertRaisesRegex(
                ValueError, (r"^all the input array dimensions except for the "
                             r"concatenation axis must match exactly, but "
                             r"along dimension 0, the array at index 0 has "
                             r"size 1 and the array at index 1 has size 3")
            ):
                dwave.optimization.symbols.Concatenate((A, B), axis)

        with self.subTest("axis out of bounds"):
            A = model.constant(np.arange(6)).reshape((1, 2, 3))
            B = model.constant(np.arange(6)).reshape((1, 2, 3))
            axis = 3
            with self.assertRaisesRegex(
                ValueError, r"^axis 3 is out of bounds for array of dimension 3"
            ):
                dwave.optimization.symbols.Concatenate((A, B), axis)


class TestConstant(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(25, dtype=np.double).reshape((5, 5)))
        B = model.constant([0, 1, 2, 3, 4])
        model.constant(np.arange(25, dtype=np.double))
        model.constant(np.arange(1, 26, dtype=np.double).reshape((5, 5)))
        model.lock()
        yield A
        yield B

    def test_truthy(self):
        model = Model()

        self.assertTrue(model.constant(1))
        self.assertFalse(model.constant(0))
        self.assertTrue(model.constant(1.1))
        self.assertFalse(model.constant(0.0))

        self.assertTrue(not model.constant(0))
        self.assertFalse(not model.constant(1))

        # these are all ambiguous
        with self.assertRaises(ValueError):
            bool(model.constant([]))
        with self.assertRaises(ValueError):
            bool(model.constant([0, 1]))
        with self.assertRaises(ValueError):
            bool(model.constant([0]))

        # the type is correct
        self.assertIsInstance(model.constant(123.4).__bool__(), bool)

    def test_comparisons(self):
        model = Model()

        # zero = model.constant(0)
        one = model.constant(1)
        onetwo = model.constant([1, 2])

        operators = [
            operator.eq,
            operator.ge,
            operator.gt,
            operator.le,
            operator.lt,
            operator.ne
        ]

        for op in operators:
            with self.subTest(op):
                self.assertEqual(op(one, 1), op(1, 1))
                self.assertEqual(op(1, one), op(1, 1))

                self.assertEqual(op(one, 2), op(1, 2))
                self.assertEqual(op(2, one), op(2, 1))

                self.assertEqual(op(one, 0), op(1, 0))
                self.assertEqual(op(0, one), op(0, 1))

                np.testing.assert_array_equal(op(onetwo, [1, 2]), op(np.asarray([1, 2]), [1, 2]))
                np.testing.assert_array_equal(op(onetwo, [1, 0]), op(np.asarray([1, 2]), [1, 0]))
                np.testing.assert_array_equal(op([1, 2], onetwo), op(np.asarray([1, 2]), [1, 2]))
                np.testing.assert_array_equal(op([1, 0], onetwo), op(np.asarray([1, 0]), [1, 2]))

    def test_copy(self):
        model = Model()

        arr = np.arange(25, dtype=np.double).reshape((5, 5))
        A = model.constant(arr)

        np.testing.assert_array_equal(A, arr)
        self.assertTrue(np.shares_memory(A, arr))

    def test_index(self):
        model = Model()

        self.assertEqual(list(range(model.constant(0))), [])
        self.assertEqual(list(range(model.constant(4))), [0, 1, 2, 3])

        with self.assertRaises(TypeError):
            range(model.constant([0]))  # not a scalar

        self.assertEqual(int(model.constant(0)), 0)
        self.assertEqual(int(model.constant(1)), 1)

        with self.assertRaises(TypeError):
            int(model.constant([0]))  # not a scalar

    def test_noncontiguous(self):
        model = Model()
        c = model.constant(np.arange(6)[::2])
        np.testing.assert_array_equal(c, [0, 2, 4])

    def test_scalar(self):
        model = Model()
        c = model.constant(5)
        self.assertEqual(c.shape(), tuple())
        np.testing.assert_array_equal(c, 5)

    def test_nonfinite_values(self):
        model = Model()

        with self.assertRaises(ValueError):
            model.constant(float("nan"))

        with self.assertRaises(ValueError):
            model.constant(float("inf"))

        with self.assertRaises(ValueError):
            model.constant(-float("inf"))

        with self.assertRaises(ValueError):
            model.constant(np.array([0, 5, np.nan]))


class TestCopy(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        c = model.constant(np.arange(25).reshape(5, 5))
        c_copy = c.copy()
        c_indexed_copy = c[::2, 1:4].copy()
        with model.lock():
            yield c_copy
            yield c_indexed_copy

    def test_simple(self):
        model = Model()
        c = model.constant(np.arange(25).reshape(5, 5))
        copy = c[::2, 1:4].copy()
        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(copy.state(), np.arange(25).reshape(5, 5)[::2, 1:4])


class TestDisjointBitSetsVariable(utils.SymbolTests):
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

    def test_construction(self):
        model = Model()

        with self.assertRaises(ValueError):
            model.disjoint_bit_sets(-5, 1)
        with self.assertRaises(ValueError):
            model.disjoint_bit_sets(1, -5)

        model.states.resize(1)

        ds, (x,) = model.disjoint_bit_sets(0, 1)
        self.assertEqual(x.shape(), (0,))

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


class TestDisjointListsVariable(utils.SymbolTests):
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

    def test_construction(self):
        model = Model()

        with self.assertRaises(ValueError):
            model.disjoint_lists(-5, 1)
        with self.assertRaises(ValueError):
            model.disjoint_lists(1, -5)

        model.states.resize(1)

        ds, (x,) = model.disjoint_lists(0, 1)
        self.assertEqual(x.shape(), (-1,))  # todo: handle this special case

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


# NOTE: Inheriting from BinaryOpTests causes runtime errors so just inheritting from SymbolTests
class TestDivide(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        ab = a / b
        ba = b / a
        model.lock()
        yield ab
        yield ba

    def test_doc_example(self):
        model = Model()
        i = model.integer(2, lower_bound=1)
        j = model.integer(2, lower_bound=1)
        k = i / j
        with model.lock():
            model.states.resize(1)
            i.set_state(0, [21, 10])
            j.set_state(0, [7, 2])
            self.assertListEqual([3., 5.], list(k.state(0)))

    def test_simple_division(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        x = a / b
        self.assertEqual(model.num_symbols(), 3)

        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(x.shape(), b.shape())
        self.assertEqual(x.size(), a.size())
        self.assertEqual(x.size(), b.size())

        model.lock()
        model.states.resize(1)
        self.assertEqual(x.state(0), 5.0/7.0)

    def test_unlike_shapes(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(ValueError):
            a / b

    def test_division_by_zero(self):
        model = Model()

        a = model.constant(5)
        b = model.constant(0)

        with self.assertRaises(ValueError):
            a / b

    def test_unsupported_floor_division(self):
        model = Model()

        a = model.constant(np.zeros((5, 5)))
        b = model.constant(np.zeros((6, 4)))

        with self.assertRaises(TypeError):
            a // b


class TestExpit(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(1.3)
        op_a = expit(a)
        model.lock()
        yield op_a

    def test_simple_inputs(self):
        model = Model()
        empty = expit(model.constant(0))
        model.lock()
        model.states.resize(1)
        self.assertEqual(empty.state(), 0.5)  # confirm consistency with SciPy expit

        simple_inputs = [-4.233307123062264, 10.342474115374873, -5.365114707829095, 0.5642821364057298]
        scipy_expit_output = [0.014296975254548053, 0.9999677665669093, 0.004655151939447702, 0.6374427639097291]
        for i, si in enumerate(simple_inputs):
            model = Model()
            expit_node = expit(model.constant(si))
            model.lock()
            model.states.resize(1)

            self.assertEqual(expit_node.state(), scipy_expit_output[i])  # confirm consistency with SciPy expit


class TestIntegerVariable(utils.SymbolTests):
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

        model.integer([10])

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


class TestLessEqual(utils.SymbolTests):
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


class TestListVariable(utils.SymbolTests):
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

    def test_construction(self):
        model = Model()

        with self.assertRaises(ValueError):
            model.list(-1)

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


class TestLog(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(1.3)
        op_a = log(a)
        model.lock()
        yield op_a

    def test_simple_inputs(self):
        model = Model()
        empty = log(model.constant(1))
        model.lock()
        model.states.resize(1)
        self.assertEqual(empty.state(), 0.0)

        simple_inputs = [1.0077188, 0.74163411, 5.06644204, 2.92553724]
        numpy_log_output = [
            0.007689162476326917, -0.29889927064260485, 1.6226388039908046, 1.0734781356130927
        ]
        for i, si in enumerate(simple_inputs):
            model = Model()
            log_node = log(model.constant(si))
            model.lock()
            model.states.resize(1)
            # confirm consistency with numpy log
            self.assertEqual(log_node.state(), numpy_log_output[i])


class TestLogical(utils.UnaryOpTests):
    def op(self, x):
        return x != 0

    def symbol_op(self, x):
        return logical(x)


class TestMax(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.max()
        b = B.max()
        model.lock()
        yield a
        yield b

    def test_empty(self):
        model = Model()
        with self.assertRaisesRegex(ValueError, "no identity"):
            model.constant([]).max()

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


class TestMaximum(utils.SymbolTests):
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


class TestMin(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.min()
        b = B.min()
        model.lock()
        yield a
        yield b

    def test_empty(self):
        model = Model()
        with self.assertRaisesRegex(ValueError, "no identity"):
            model.constant([]).min()

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


class TestMinimum(utils.SymbolTests):
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


class TestModule(unittest.TestCase):
    def test__all__(self):
        # make sure every name in __all__ actually exists
        for name in dwave.optimization.symbols.__all__:
            getattr(dwave.optimization.symbols, name)

        for name in dwave.optimization.mathematical.__all__:
            getattr(dwave.optimization.mathematical, name)


class TestModulus(utils.BinaryOpTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(-5)
        c = model.constant(3)
        d = model.constant(-3)
        ac = a % c
        ad = a % d
        bc = b % c
        bd = b % d
        with model.lock():
            yield from (ac, ad, bc, bd)

    def op(self, lhs, rhs):
        return np.mod(lhs, rhs)

    def symbol_op(self, lhs, rhs):
        return lhs % rhs

    def test_function_operator_equivalence(self):
        model = Model()
        i = model.constant([5, -5, 5, -5])
        j = model.constant([3, 3, -3, -3])

        mod_1 = i % j
        mod_2 = mod(i, j)
        with model.lock():
            model.states.resize(1)
            np.testing.assert_array_equal(mod_1.state(0), mod_2.state(0))

    def test_scalar_mod(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(-5)
        c = model.constant(3)
        d = model.constant(-3)
        ac = a % c
        ad = a % d
        bc = b % c
        bd = b % d

        with model.lock():
            model.states.resize(1)
            self.assertEqual(ac.state(0), 2)
            self.assertEqual(ad.state(0), -1)
            self.assertEqual(bc.state(0), 1)
            self.assertEqual(bd.state(0), -2)

    def test_array_mod(self):
        model = Model()
        a = model.constant([5, -5, 5, -5])
        b = model.constant([3, 3, -3, -3])
        ab = a % b

        with model.lock():
            model.states.resize(1)
            np.testing.assert_array_equal(ab.state(0), [2, 1, -1, -2])

    def test_float_mod(self):
        model = Model()
        a = model.constant([5, -5, 5, -5])
        b = model.constant([3.33, 3.33, -3.33, -3.33])
        ab = a % b

        with model.lock():
            model.states.resize(1)
            np.testing.assert_allclose(ab.state(0), [1.67, 1.66, -1.66, -1.67], rtol=1e-10)

    def test_zero_mod(self):
        model = Model()
        values: typing.Union[int, list[int]] = [
            0,
            1,
            -1,
            [1, -2, 3],
            [0, -1, 2],
        ]
        for lhs, rhs in itertools.product(values, repeat=2):
            with np.errstate(divide='ignore'):
                np_result = np.mod(lhs, rhs)

            lhs_c = model.constant(lhs)
            rhs_c = model.constant(rhs)
            result = lhs_c % rhs_c

            with model.lock():
                model.states.resize(1)
                np.testing.assert_equal(np_result, result.state(0))


class TestMultiply(utils.SymbolTests):
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


class TestNaryAdd(utils.NaryOpTests):
    def op(self, *xs):
        return sum(xs)

    def node_op(self, *xs):
        return dwave.optimization.add(*xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryAdd

    def test_iadd(self):
        model = Model()
        x: dwave.optimization.model.ArraySymbol = model.binary()  # typing is for mypy
        a = model.constant(5)
        b = model.constant(4)

        x += a  # type promotion to a NaryAdd
        self.assertIsInstance(x, dwave.optimization.symbols.NaryAdd)

        y = x
        x += b
        self.assertIs(x, y)  # subsequent should be in-place

    def test_mismatched_shape(self):
        model = Model()
        x: dwave.optimization.model.ArraySymbol = model.binary()  # typing is for mypy
        a = model.constant(0)

        b = model.constant([0, 1])

        with self.assertRaises(ValueError):
            x += b  # before promotion

        x += a  # get a NaryAdd
        self.assertIsInstance(x, dwave.optimization.symbols.NaryAdd)

        with self.assertRaises(ValueError):
            x += b  # after promotion


class TestNaryMaximum(utils.NaryOpTests):
    def op(self, *xs):
        return max(xs)

    def node_op(self, *xs):
        return dwave.optimization.maximum(*xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryMaximum


class TestNaryMinimum(utils.NaryOpTests):
    def op(self, *xs):
        return min(xs)

    def node_op(self, *xs):
        return dwave.optimization.minimum(*xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryMinimum


class TestNaryMultiply(utils.NaryOpTests):
    def op(self, *xs):
        return math.prod(xs)

    def node_op(self, *xs):
        return dwave.optimization.multiply(*xs)

    def node_class(self):
        return dwave.optimization.symbols.NaryMultiply

    def test_imul(self):
        model = Model()
        x: dwave.optimization.model.ArraySymbol = model.binary()  # typing is for mypy
        a = model.constant(5)
        b = model.constant(4)

        x *= a  # type promotion to a NaryAdd
        self.assertIsInstance(x, dwave.optimization.symbols.NaryMultiply)

        y = x
        x *= b
        self.assertIs(x, y)  # subsequent should be in-place

    def test_mismatched_shape(self):
        model = Model()
        x: dwave.optimization.model.ArraySymbol = model.binary()  # typing is for mypy
        a = model.constant(0)

        b = model.constant([0, 1])

        with self.assertRaises(ValueError):
            x *= b  # before promotion

        x *= a  # get a NaryMultiply
        self.assertIsInstance(x, dwave.optimization.symbols.NaryMultiply)

        with self.assertRaises(ValueError):
            x *= b  # after promotion


class TestNegate(utils.UnaryOpTests):
    def op(self, x):
        return -x


class TestNot(utils.UnaryOpTests):
    def op(self, x):
        return np.logical_not(x)

    def symbol_op(self, x):
        return logical_not(x)


class TestOr(utils.BinaryOpTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(1)
        b = model.constant(1)
        c = model.constant(0)
        d = model.constant(0)
        ab = logical_or(a, b)
        ac = logical_or(a, c)
        cd = logical_or(c, d)
        cb = logical_or(c, b)
        with model.lock():
            yield from (ab, ac, cd, cb)

    def op(self, lhs, rhs):
        return np.logical_or(lhs, rhs)

    def symbol_op(self, lhs, rhs):
        return logical_or(lhs, rhs)

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
            a | b
        with self.assertRaises(AttributeError):
            a.logical_or(b)

    def test_array_or(self):
        model = Model()
        a = model.constant([1, 0, 1, 0])
        b = model.constant([1, 1, 1, 1])
        c = model.constant([0, 0, 0, 0])
        ab = logical_or(a, b)
        ac = logical_or(a, c)

        model.lock()
        model.states.resize(1)

        np.testing.assert_array_equal(ab.state(0), [1, 1, 1, 1])
        np.testing.assert_array_equal(ac.state(0), [1, 0, 1, 0])


class TestPartialProd(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(8).reshape((2, 2, 2)))
        B = model.constant(np.arange(25).reshape((5, 5)))
        a = A.prod(axis=0)
        b = B.prod(axis=1)
        model.lock()
        yield a
        yield b

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(8).reshape((2, 2, 2)))
        B = model.constant(np.arange(25).reshape((5, 5)))
        a = A.prod(axis=0)
        b = B.prod(axis=1)
        model.lock()
        model.states.resize(1)
        np.testing.assert_array_equal(a.state(0), np.prod(np.arange(8).reshape((2, 2, 2)), axis=0))
        np.testing.assert_array_equal(b.state(0), np.prod(np.arange(25).reshape((5, 5)), axis=1))

    def test_indexed(self):
        model = Model()
        A = model.constant(np.arange(125).reshape((5, 5, 5)))
        x = model.list(5)
        ax = A[:, x, :].prod(axis=1)
        model.lock()
        model.states.resize(1)
        x.set_state(0, [0, 4, 1, 3, 2])
        np.testing.assert_array_equal(
            ax.state(0),
            np.prod(np.arange(125).reshape((5, 5, 5))[:, [0, 4, 1, 3, 2], :], axis=1)
        )


class TestPartialSum(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(8).reshape((2, 2, 2)))
        B = model.constant(np.arange(25).reshape((5, 5)))
        a = A.sum(axis=0)
        b = B.sum(axis=1)
        model.lock()
        yield a
        yield b

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(8).reshape((2, 2, 2)))
        B = model.constant(np.arange(25).reshape((5, 5)))
        a = A.sum(axis=0)
        b = B.sum(axis=1)
        model.lock()
        model.states.resize(1)
        np.testing.assert_array_equal(a.state(0), np.sum(np.arange(8).reshape((2, 2, 2)), axis=0))
        np.testing.assert_array_equal(b.state(0), np.sum(np.arange(25).reshape((5, 5)), axis=1))

    def test_indexed(self):
        model = Model()
        A = model.constant(np.arange(125).reshape((5, 5, 5)))
        x = model.list(5)
        ax = A[:, x, :].sum(axis=1)
        model.lock()
        model.states.resize(1)
        x.set_state(0, [0, 4, 1, 3, 2])
        np.testing.assert_array_equal(
            ax.state(0),
            np.sum(np.arange(125).reshape((5, 5, 5))[:, [0, 4, 1, 3, 2], :], axis=1)
        )


class TestPermutation(utils.SymbolTests):
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


class TestProd(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.prod()
        b = B.prod()
        model.lock()
        yield a
        yield b

    def test_empty(self):
        model = Model()
        empty = model.constant([]).prod()
        model.lock()
        model.states.resize(1)

        self.assertEqual(empty.state(), 1)
        self.assertEqual(empty.state(), np.asarray([]).prod())  # confirm consistency with NumPy

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


class TestPut(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()

        array = model.constant([0, 1, 2])
        indices = model.constant([1])
        values = model.constant([-1])
        p = put(array, indices, values)
        with model.lock():
            yield p

    def test_1d(self):
        model = Model()

        array = model.constant(np.zeros((3, 3)))
        indices = model.constant([0, 1, 2])
        values = model.integer(3)
        array = put(array, indices, values)

        model.states.resize(1)
        with model.lock():
            values.set_state(0, [10, 20, 30])
            np.testing.assert_array_equal(array.state(), [[10, 20, 30], [0, 0, 0], [0, 0, 0]])
            values.set_state(0, [100, 200, 300])
            np.testing.assert_array_equal(array.state(), [[100, 200, 300], [0, 0, 0], [0, 0, 0]])

    def test_disallow_scalar(self):
        model = Model()

        array = model.constant(0)
        indices = model.constant([0])
        values = model.constant([1])

        with self.assertRaises(ValueError):
            put(array, indices, values)

    def test_disallow_different_shapes(self):
        model = Model()

        array = model.constant(np.zeros((3, 3)))
        indices = model.set(3)
        values = model.set(3)  # may be a different size than indices
        with self.assertRaises(ValueError):
            put(array, indices, values)

    def test_disallow_out_of_range(self):
        model = Model()

        array = model.constant(np.zeros((3, 3)))
        indices = model.integer(3, upper_bound=100)  # may be out of bounds
        values = model.constant([0, 1, 2])
        with self.assertRaises(IndexError):
            put(array, indices, values)


class TestQuadraticModel(utils.SymbolTests):
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


class TestReshape(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(12))
        syms = [A.reshape(12), A.reshape((2, 6)), A.reshape(3, 4)]
        model.lock()
        yield from syms

    def test_implicit_reshape(self):
        model = Model()
        A = model.constant(np.arange(12).reshape(3, 4))
        B = A.reshape(2, -1)
        C = A.reshape(-1, 6)
        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(B.state(), np.arange(12).reshape(2, 6))
            np.testing.assert_array_equal(C.state(), np.arange(12).reshape(2, 6))

    def test_flatten(self):
        model = Model()
        A = model.constant(np.arange(25).reshape(5, 5))
        B = A.flatten()
        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(B.state(), np.arange(25))

    def test_noncontiguous(self):
        model = Model()
        A = model.constant(np.arange(20).reshape(4, 5))
        B = A[::2, :]  # 2x5 array
        C = B.reshape(5, 2)
        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(
                C.state(),
                np.arange(20).reshape(4, 5)[::2, :].reshape(5, 2),
            )


class TestRint(utils.SymbolTests):
    rng = np.random.default_rng(1)

    def generate_symbols(self):
        model = Model()
        a = model.constant(1.3)
        op_a = rint(a)
        model.lock()
        yield op_a

    def test_simple_inputs(self):
        model = Model()
        empty = rint(model.constant(0))
        model.lock()
        model.states.resize(1)
        self.assertEqual(empty.state(), np.rint(0))  # confirm consistency with NumPy

        simple_inputs = (self.rng.random(size=(100)) * 2 - 1) * 1000000
        for si in simple_inputs:
            model = Model()
            rint_node = rint(model.constant(si))
            model.lock()
            model.states.resize(1)

            self.assertEqual(rint_node.state(), np.rint(si))  # confirm consistency with NumPy

    def test_indexing_using_rint(self):
        model = Model()
        rint_node_idx3 = rint(model.constant(3.1))
        rint_node_idx6 = rint(model.constant(5.8))
        rint_node_idx8 = rint(model.constant(8))

        arr = model.constant(2 * np.arange(10))  # value at each index is (index * 2)
        six = arr[rint_node_idx3]
        twelve = arr[rint_node_idx6]
        sixteen = arr[rint_node_idx8]

        model.lock()
        model.states.resize(1)
        self.assertEqual(six.state(0), 6)
        self.assertEqual(twelve.state(0), 12)
        self.assertEqual(sixteen.state(0), 16)


class TestSquare(utils.UnaryOpTests):
    def op(self, x):
        return x ** 2


class TestSquareRoot(utils.SymbolTests):
    rng = np.random.default_rng(1)

    def generate_symbols(self):
        model = Model()
        a = model.constant(125)
        op_a = sqrt(a)
        model.lock()
        yield op_a

    def test_simple_inputs(self):
        model = Model()
        empty = sqrt(model.constant(0))
        model.lock()
        model.states.resize(1)

        self.assertEqual(empty.state(), np.sqrt(0))  # confirm consistency with NumPy

        simple_inputs = self.rng.random(size=(100)) * 1000000
        for si in simple_inputs:
            model = Model()
            sqrt_node = sqrt(model.constant(si))
            model.lock()
            model.states.resize(1)

            self.assertEqual(sqrt_node.state(), np.sqrt(si))  # confirm consistency with NumPy

    def test_negative_inputs(self):
        model = Model()
        neg_one = model.constant(-1)
        self.assertRaises(ValueError, sqrt, neg_one)


class TestSetVariable(utils.SymbolTests):
    def generate_symbols(self) -> collections.abc.Iterator[dwave.optimization.symbols.SetVariable]:
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


class TestSize(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()

        a = dwave.optimization.symbols.Size(model.constant(5))
        b = dwave.optimization.symbols.Size(model.constant([0, 1, 2]))
        c = model.set(5).size()

        with model.lock():
            yield a
            yield b
            yield c

    def test_dynamic(self):
        model = Model()
        model.states.resize(2)

        set_ = model.set(5)
        length = set_.size()

        set_.set_state(0, [])
        set_.set_state(1, [0, 2, 3])

        with model.lock():
            self.assertEqual(length.state(0), 0)
            self.assertEqual(length.state(1), 3)

    def test_scalar(self):
        model = Model()
        model.states.resize(1)

        length = dwave.optimization.symbols.Size(model.constant(1))

        with model.lock():
            self.assertEqual(length.state(), 1)


class TestSubtract(utils.BinaryOpTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(5)
        b = model.constant(7)
        x = a - b
        y = b - a
        with model.lock():
            yield x
            yield y

    def op(self, lhs, rhs):
        return lhs - rhs

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


class TestSum(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.sum()
        b = B.sum()
        model.lock()
        yield a
        yield b

    def test_empty(self):
        model = Model()
        empty = model.constant([]).sum()
        model.lock()
        model.states.resize(1)

        self.assertEqual(empty.state(), 0)
        self.assertEqual(empty.state(), np.asarray([]).sum())  # confirm consistency with NumPy

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(5))
        B = model.constant(np.arange(5, 10))
        a = A.sum()
        b = B.sum()
        model.states.resize(1)
        model.lock()

        self.assertEqual(a.state(0), np.arange(5).sum())
        self.assertEqual(b.state(0), np.arange(5, 10).sum())


class TestWhere(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        condition = model.binary()
        x = model.constant([1, 2, 3])
        y = model.constant([10, 20, 30])
        where = dwave.optimization.where(condition, x, y)
        with model.lock():
            yield where

    def test_dynamic(self):
        model = Model()
        condition = model.binary()
        x = model.set(10)  # any subset of range(10)
        y = model.set(10)  # any subset of range(10)
        where = dwave.optimization.where(condition, x, y)
        with model.lock():
            model.states.resize(1)
            condition.set_state(0, True)
            x.set_state(0, [0., 2., 3.])
            y.set_state(0, [1])
            np.testing.assert_array_equal(where.state(), [0, 2, 3])
            condition.set_state(0, False)
            np.testing.assert_array_equal(where.state(), [1])

        with self.assertRaises(ValueError):
            # wrong shape
            dwave.optimization.where(model.binary(3), x, y)


class TestXor(utils.BinaryOpTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(1)
        b = model.constant(1)
        c = model.constant(0)
        d = model.constant(0)
        ab = logical_xor(a, b)
        ac = logical_xor(a, c)
        cd = logical_xor(c, d)
        cb = logical_xor(c, b)
        with model.lock():
            yield from (ab, ac, cd, cb)

    def op(self, lhs, rhs):
        return np.logical_xor(lhs, rhs)

    def symbol_op(self, lhs, rhs):
        return logical_xor(lhs, rhs)

    def test_scalar_xor(self):
        model = Model()
        a = model.constant(1)
        b = model.constant(1)
        c = model.constant(0)
        d = model.constant(0)
        ab = logical_xor(a, b)
        ac = logical_xor(a, c)
        cd = logical_xor(c, d)
        cb = logical_xor(c, b)
        self.assertEqual(model.num_symbols(), 8)

        model.lock()
        model.states.resize(1)
        self.assertEqual(ab.state(0), 0)
        self.assertEqual(ac.state(0), 1)
        self.assertEqual(cd.state(0), 0)
        self.assertEqual(cb.state(0), 1)

        with self.assertRaises(TypeError):
            a ^ b
        with self.assertRaises(AttributeError):
            a.logical_xor(b)

    def test_array_xor(self):
        model = Model()
        a = model.constant([1, 0, 1, 0])
        b = model.constant([1, 1, 1, 1])
        c = model.constant([0, 0, 0, 0])
        ab = logical_xor(a, b)
        ac = logical_xor(a, c)

        model.lock()
        model.states.resize(1)

        np.testing.assert_array_equal(ab.state(0), [0, 1, 0, 1])
        np.testing.assert_array_equal(ac.state(0), [1, 0, 1, 0])
