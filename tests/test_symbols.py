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
import sys
import typing
import unittest

import numpy as np

import dwave.optimization
from dwave.optimization.mathematical import softmax
import dwave.optimization.symbols
from dwave.optimization import (
    Model,
    arange,
    broadcast_to,
    bspline,
    exp,
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
    safe_divide,
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

    def test_absolute(self):
        from dwave.optimization.symbols import Absolute

        model = Model()
        x = model.integer(5, lower_bound=-5, upper_bound=5)
        a = dwave.optimization.absolute(x)
        self.assertIsInstance(a, Absolute)
        self.assertEqual(model.num_symbols(), 2)

        model.states.resize(1)
        with model.lock():
            x.set_state(0, [-2, -1, 0, 1, 5])
            np.testing.assert_array_equal(a.state(), [2, 1, 0, 1, 5])


class TestAccumulateZip(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        c0 = model.constant([0, 0])
        c1 = model.constant([0, 1])

        @dwave.optimization.expression
        def expr(a, b, c):
            return a + b + c

        acc = dwave.optimization.symbols.AccumulateZip(expr, (c0, c1), initial=7)
        acc_with_initial_node = dwave.optimization.symbols.AccumulateZip(
            expr, (c0, c1), initial=model.constant(7)
        )

        with model.lock():
            yield acc
            yield acc_with_initial_node

    def test_mismatched_inputs(self):
        model = Model()
        c0 = model.constant([0, 0])

        @dwave.optimization.expression
        def expr(a, b, c):
            return a + b + c

        with self.assertRaises(ValueError):
            dwave.optimization.symbols.AccumulateZip(expr, (c0,))

    def test_invalid_expression_non_scalar(self):
        model = Model()
        c0 = model.constant([0, 0])

        # Can't use an Expression that uses a non-scalar input
        # In the future we may prevent this when forming the expression
        expr = dwave.optimization.expression(
            lambda a, b: a + b,
            a=dict(lower_bound=-10, upper_bound=10, integral=False, shape=(1,)),
        )

        with self.assertRaises(ValueError):
            dwave.optimization.symbols.AccumulateZip(expr, (c0,))

    def test_invalid_expression_input_range(self):
        model = Model()
        c0 = model.constant([0, 0])

        @dwave.optimization.expression(inp0=dict(lower_bound=0, upper_bound=100),
                                       inp2=dict(lower_bound=0, upper_bound=10))
        def expr(inp0, inp1):
            return inp0 + inp1

        with self.assertRaises(ValueError):
            dwave.optimization.symbols.AccumulateZip(expr, (c0,))


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

    def test_constant_promotion(self):
        model = Model()
        x = model.integer((2, 3, 5))
        x[1, [0, 2], model.constant([0, 3])]

        with self.assertRaisesRegex(
                IndexError,
                "index's largest possible value 3 is out of bounds for axis 0 with size 3"):
            x[1, [0, 3], model.constant([0, 3])]

        with self.assertRaisesRegex(
                IndexError,
                "index may not contain non-integer values for axis 0"):
            x[1, [0, 1.1], model.constant([0, 3])]

        with self.assertRaisesRegex(IndexError, "only integers, slices"):
            x[1, [0, float("inf")], model.constant([0, 3])]


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
        b = model.constant(2)
        c = model.constant(0)
        ab = logical_and(a, b)
        ac = logical_and(a, c)
        cc = logical_and(c, c)
        cb = logical_and(c, b)
        self.assertEqual(model.num_symbols(), 7)

        model.lock()

        yield from (ab, ac, cc, cb)

    def op(self, lhs, rhs):
        return np.logical_and(lhs, rhs)

    def symbol_op(self, lhs, rhs):
        return logical_and(lhs, rhs)

    def test_scalar_and(self):
        model = Model()
        a = model.constant(1)
        b = model.constant(2)
        c = model.constant(0)
        ab = logical_and(a, b)
        ac = logical_and(a, c)
        cc = logical_and(c, c)
        cb = logical_and(c, b)
        self.assertEqual(model.num_symbols(), 7)

        model.lock()
        model.states.resize(1)
        self.assertEqual(ab.state(0), 1)
        self.assertEqual(ac.state(0), 0)
        self.assertEqual(cc.state(0), 0)
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


class TestArgSort(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        c = model.constant([1, 5, 2, 3])
        argsort = dwave.optimization.symbols.ArgSort(c)

        with model.lock():
            yield argsort

    def test_indexing(self):
        model = Model()
        s = model.set(5)
        c = model.constant(range(5))
        a = dwave.optimization.mathematical.argsort(s)
        b = c[a]

        with model.lock():
            model.states.resize(1)
            s.set_state(0, [1, 4, 2])
            np.testing.assert_array_equal(b.state(), [0, 2, 1])


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

    def test_bounds(self):
        model = Model()
        x = model.binary(lower_bound=0, upper_bound=1)
        self.assertEqual(x.lower_bound(), 0)
        self.assertEqual(x.upper_bound(), 1)

        x = model.binary((2, 2), upper_bound=0)
        self.assertTrue(x.upper_bound() == 0)

        x = model.binary((2, 3), -3, [[1, 0, 0], [1, 0, 0]])
        self.assertEqual(x.lower_bound(), 0.0)
        self.assertTrue(np.all(x.upper_bound() == [[1, 0, 0], [1, 0, 0]]))

        with self.assertRaises(ValueError):
            model.integer((2, 3), upper_bound=np.nan)

        with self.assertRaises(ValueError):
            model.integer((2, 3), upper_bound=np.arange(6))

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

    def test_serialization(self):
        model = Model()
        binary_vars = [
            model.binary((5, 2)),
            model.binary(),
            model.binary(3, lower_bound=1),
            model.binary(2, upper_bound=[0,1]),
        ]

        model.lock()
        with model.to_file() as f:
            copy = Model.from_file(f)

        for old, new in zip(binary_vars, copy.iter_decisions()):
            self.assertEqual(old.shape(), new.shape())
            for i in range(old.size()):
                self.assertTrue(np.all(old.lower_bound() == new.lower_bound()))
                self.assertTrue(np.all(old.upper_bound() == new.upper_bound()))

    def test_set_state(self):
        with self.subTest("array-like"):
            model = Model()
            model.states.resize(1)
            x = model.binary([5, 5])

            x.set_state(0, np.arange(25) % 2)
            np.testing.assert_array_equal(x.state(), np.arange(25).reshape((5, 5)) % 2)
            x.set_state(0, 1 - np.arange(25).reshape((5, 5)) % 2)
            np.testing.assert_array_equal(x.state(), 1 - np.arange(25).reshape((5, 5)) % 2)

        with self.subTest("Default bounds test"):
            model = Model()
            model.states.resize(1)
            x = model.binary(1)
            x.set_state(0, 0)
            with np.testing.assert_raises(ValueError):
                x.set_state(0, -1)
            with np.testing.assert_raises(ValueError):
                x.set_state(0, 2)

        with self.subTest("Simple bounds test"):
            model = Model()
            model.states.resize(1)
            x = model.binary(2, lower_bound=[-1, 0.9], upper_bound=[1.1, 1.2])
            x.set_state(0, [0, 1])
            with np.testing.assert_raises(ValueError):
                x.set_state(0, 2)
            with np.testing.assert_raises(ValueError):
                x.set_state(1, 0)

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


class TestBroadcastTo(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        x = model.constant(np.asarray([[0], [1], [2]]))
        b0 = broadcast_to(x, (3, 4))
        b1 = broadcast_to(x, (2, 3, 1))
        y = model.constant(5)
        b2 = broadcast_to(y, 10)
        with model.lock():
            yield from [b0, b1, b2]

    def test_exceptions(self):
        model = Model()
        x = model.integer(5)
        with self.assertRaisesRegex(
            ValueError,
            r"array of shape \(5,\) could not be broadcast to \(3,\)",
        ):
            broadcast_to(x, 3)
        with self.assertRaisesRegex(
            ValueError,
            r"array of shape \(5,\) could not be broadcast to \(5, 1\)",
        ):
            broadcast_to(x, (5, 1))

        with self.assertRaises(ValueError):
            broadcast_to(x, "hello")

        self.assertEqual(model.num_symbols(), 1)  # no side effects

    def test_state_size(self):
        for symbol in self.generate_symbols():
            self.assertEqual(symbol.state_size(), 0)

        model = Model()
        x = model.constant(5)
        dwave.optimization.broadcast_to(x, (10, 5))
        self.assertEqual(model.num_symbols(), 2)
        self.assertEqual(model.state_size(), x.state_size())


class TestBSpline(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        x = model.constant([2])
        k = 2
        t = [0, 1, 2, 3, 4, 5, 6]
        c = [-1, 2, 0, -1]
        bspline_node = bspline(x, k, t, c)
        model.lock()
        yield bspline_node

    def test_simple(self):
        model = Model()
        x = model.constant([2.0, 2.5, 3.0, 3.5, 4.0])
        k = 2
        t = [0, 1, 2, 3, 4, 5, 6]
        c = [-1, 2, 0, -1]
        bspline_node = bspline(x, k, t, c)
        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(bspline_node.state(),
                                          [0.5, 1.375, 1.0, 0.125, -0.5])

    def test_errors(self):
        model = Model()
        with self.subTest("node pointer does not accept multi-d arrays"):
            x = model.constant([[0, 2.0], [2.5, 3.0]])
            k = 2
            t = [0, 1, 2, 3, 4, 5, 6]
            c = [-1, 2, 0, -1]
            with self.assertRaisesRegex(
                    ValueError, ("node pointer cannot be multi-d array")
            ):
                bspline(x, k, t, c)
        with self.subTest("degree is less than 5"):
            x = model.constant([2.0, 2.5, 3.0, 3.5, 4.0])
            k = 5
            t = [0, 1, 2, 3, 4, 5, 6]
            c = [-1, 2, 0, -1]
            with self.assertRaisesRegex(
                    ValueError, ("bspline degree should be smaller than 5")
            ):
                bspline(x, k, t, c)
        with self.subTest("number of knots is smaller than 20"):
            x = model.constant([2.0, 2.5, 3.0, 3.5, 4.0])
            k = 4
            t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 15, 16, 17, 18, 19]
            c = [-1, 2, 0, -1]
            with self.assertRaisesRegex(
                    ValueError, ("number of knots should be smaller than 20")
            ):
                bspline(x, k, t, c)
        with self.subTest("degree, len(knots) and len(coefficients)"):
            x = model.constant([2.0, 2.5, 3.0, 3.5, 4.0])
            k = 2
            t = [0, 1, 2, 3, 4, 5]
            c = [-1, 2, 0, -1]
            with self.assertRaisesRegex(
                    ValueError, ("number of knots should be equal to sum of"
                                 "degree, number of coefficients and 1")
            ):
                bspline(x, k, t, c)
        with self.subTest("bspline node only interpolates inside the base interval"):
            x = model.constant([0, 5])
            k = 2
            t = [0, 1, 2, 3, 4, 5, 6]
            c = [-1, 2, 0, -1]
            with self.assertRaisesRegex(
                    ValueError, ("bspline node only interpolates inside the base interval: "
                                 "2.000000 to 4.000000")
            ):
                bspline(x, k, t, c)

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

    @unittest.skipIf((sys.version_info.major, sys.version_info.minor) < (3, 12),
                     "Python-level access to the buffer protocol requires Python 3.12+")
    def test_buffer_flags(self):
        import inspect  # for the buffer flags

        model = Model()

        A = model.constant(np.arange(25).reshape(5, 5))

        # A few smoke tests to make sure we're not being overly restrictive with
        # our flags for the cases we care about
        memoryview(A)
        np.asarray(A)

        # We never allow a writeable buffer
        with self.assertRaises(BufferError):
            A.__buffer__(inspect.BufferFlags.WRITABLE)

        # While we often incidentally export contiguous buffers, we don't
        # respect the request (for now)
        with self.assertRaises(BufferError):
            A.__buffer__(inspect.BufferFlags.ANY_CONTIGUOUS)
        with self.assertRaises(BufferError):
            A.__buffer__(inspect.BufferFlags.C_CONTIGUOUS)
        with self.assertRaises(BufferError):
            A.__buffer__(inspect.BufferFlags.F_CONTIGUOUS)

        # Currently we always expose stride information
        with self.assertRaises(BufferError):
            A.__buffer__(inspect.BufferFlags.ND)
        with self.assertRaises(BufferError):
            A.__buffer__(inspect.BufferFlags.SIMPLE)

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

    def test_readonly(self):
        model = Model()

        arr = np.ones(3)
        arr.setflags(write=False)
        c = model.constant(arr)
        np.testing.assert_array_equal(arr, c)

    def test_interning(self):
        model = Model()

        c = model.constant(5)
        self.assertEqual(model.num_symbols(), 1)

        d = model.constant(5)
        self.assertEqual(model.num_symbols(), 1)
        self.assertEqual(c.id(), d.id())

        d = model.constant([5])
        self.assertEqual(model.num_symbols(), 2)
        self.assertNotEqual(c.id(), d.id())

        a = model.constant([1, 2, 3, 4])
        self.assertEqual(model.num_symbols(), 3)

        b = model.constant([1, 2, 3, 4])
        self.assertEqual(model.num_symbols(), 3)
        self.assertEqual(a.id(), b.id())

        x = model.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(model.num_symbols(), 4)
        self.assertEqual(x.shape(), (2, 4))

        y = model.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
        self.assertEqual(model.num_symbols(), 4)
        self.assertEqual(y.shape(), (2, 4))
        self.assertEqual(x.id(), y.id())

        z = model.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertEqual(model.num_symbols(), 5)
        self.assertEqual(z.shape(), (4, 2))
        self.assertNotEqual(x.id(), z.id())

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


class TestCos(utils.UnaryOpTests):
    def op(self, x):
        return np.cos(x)

    def symbol_op(self, x):
        return dwave.optimization.cos(x)


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


class TestExp(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        a = model.constant(1.3)
        op_a = exp(a)
        model.lock()
        yield op_a

    def test_simple_inputs(self):
        model = Model()
        empty = exp(model.constant(0))
        model.lock()
        model.states.resize(1)
        self.assertEqual(empty.state(), 1.0)

        # confirm consistency with numpy exp
        simple_inputs = [-1.0, 1.23, 3.14]
        numpy_outputs = [0.36787944117144233, 3.4212295362896734, 23.103866858722185]
        for i, si in enumerate(simple_inputs):
            model = Model()
            exp_node = exp(model.constant(si))
            model.lock()
            model.states.resize(1)
            self.assertEqual(exp_node.state(), numpy_outputs[i])


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


class TestExtract(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        condition = model.binary(3)
        arr = model.constant([1, 2, 3])
        extract = dwave.optimization.extract(condition, arr)
        with model.lock():
            yield extract

    def test(self):
        model = Model()
        condition = model.binary(3)
        arr = model.integer(3)
        extract = dwave.optimization.extract(condition, arr)
        with model.lock():
            model.states.resize(1)
            condition.set_state(0, [False, True, True])
            arr.set_state(0, [1, 2, 3])
            np.testing.assert_array_equal(extract.state(), [2, 3])
            condition.set_state(0, [False, False, True])
            np.testing.assert_array_equal(extract.state(), [3])

        with self.assertRaises(ValueError):
            # wrong shape
            dwave.optimization.extract(model.binary(4), arr)


class TestInput(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        inp = model.input(lower_bound=-10, upper_bound=10, integral=False)
        inp10 = model.input((10,))
        model.lock()

        yield inp
        yield inp10

    def test_set_state(self):
        model = Model()
        inp = dwave.optimization.symbols.Input(model, shape=(2, 1, 2))
        model.lock()

        model.states.resize(1)

        inp.set_state(0, [[[0, 1]], [[2, 3]]])

        np.testing.assert_array_equal(inp.state(), [[[0, 1]], [[2, 3]]])

        with self.assertRaises(ValueError):
            inp.set_state(0, [0, 1, 2, 3])

    def test_state_serialization(self):
        for version in dwave.optimization._model.KNOWN_SERIALIZATION_VERSIONS:
            if version < self.MIN_SERIALIZATION_VERSION:
                continue
            with self.subTest(version=version):
                model = Model()
                inp = model.input(lower_bound=-10, upper_bound=10, integral=False)
                model.lock()

                model.states.resize(1)

                # ensure serialization works if no state is set
                with model.states.to_file(version=version) as f:
                    model.states.clear()
                    model.states.from_file(f)

                # ensure serialization saves the state if set
                inp.set_state(0, -7)

                self.assertTrue(inp.has_state(0))
                self.assertEqual(inp.state(), -7)

                with model.states.to_file(version=version) as f:
                    model.states.clear()
                    model.states.from_file(f)

                self.assertEqual(inp.state(), -7)

                # test with a larger shape
                model = Model()
                inp = dwave.optimization.symbols.Input(model, shape=(2, 1, 2))
                model.lock()

                model.states.resize(1)

                inp.set_state(0, [[[0, 1]], [[2, 3]]])

                self.assertTrue(inp.has_state(0))

                with model.states.to_file(version=version) as f:
                    model.states.clear()
                    model.states.from_file(f)

                np.testing.assert_array_equal(inp.state(), [[[0, 1]], [[2, 3]]])

    def test_initializing_unset_state(self):
        # ensure proper error is raised when initializing the model state without having
        # set the input's state
        model = Model()
        inp = model.input()
        x = model.binary()
        model.minimize(inp + x)

        model.lock()

        model.states.resize(1)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^InputNode must have state explicitly initialized"
        ):
            model.objective.state()

    def test_bounds_and_integrality(self):
        model = Model()

        i0 = model.input()
        default_lower_bound = i0.lower_bound()
        default_upper_bound = i0.upper_bound()
        default_integral = i0.integral()

        i1 = model.input(lower_bound=-10)
        self.assertEqual(i1.lower_bound(), -10)
        self.assertEqual(i1.upper_bound(), default_upper_bound)
        self.assertEqual(i1.integral(), default_integral)

        i2 = model.input(upper_bound=100)
        self.assertEqual(i2.lower_bound(), default_lower_bound)
        self.assertEqual(i2.upper_bound(), 100)
        self.assertEqual(i2.integral(), default_integral)

        i3 = model.input(integral=True)
        self.assertEqual(i3.lower_bound(), default_lower_bound)
        self.assertEqual(i3.upper_bound(), default_upper_bound)
        self.assertEqual(i3.integral(), True)

        with self.assertRaises(ValueError):
            model.input(lower_bound=100, upper_bound=-100)
        with self.assertRaises(ValueError):
            model.input(lower_bound=.1, upper_bound=.9, integral=True)


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

        x = model.integer((2, 2), upper_bound=7)
        self.assertTrue(x.upper_bound() == 7)

        x = model.integer((2, 3), -3, [[1, 2, 3], [4, 5, 6]])
        self.assertTrue(np.all(x.upper_bound() == [[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(x.lower_bound() == -3)

        with self.assertRaises(ValueError):
            model.integer((2, 3), upper_bound=np.nan)

        with self.assertRaises(ValueError):
            model.integer((2, 3), upper_bound=np.arange(6))

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
            model.integer(2, lower_bound=[1, 2], upper_bound=[3, 4]),
        ]

        model.lock()
        with model.to_file() as f:
            copy = Model.from_file(f)

        for old, new in zip(integers, copy.iter_decisions()):
            self.assertEqual(old.shape(), new.shape())
            for i in range(old.size()):
                self.assertTrue(np.all(old.lower_bound() == new.lower_bound()))
                self.assertTrue(np.all(old.upper_bound() == new.upper_bound()))

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
            with self.assertRaisesRegex(ValueError, r"^values must be a subset of range\(5\)$"):
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


class TestLinearProgram(utils.SymbolTests):
    MIN_SERIALIZATION_VERSION = (1, 0)

    def generate_symbols(self):
        # test serialization on a few different scenarios of different arguments
        # to the LinearProgram symbol

        # just c
        model = Model()
        c = model.integer(2, lower_bound=-10, upper_bound=10)
        res = dwave.optimization.linprog(c)
        feasible = res.success
        obj = res.fun
        sol = res.x
        model.lock()

        yield res.lp
        yield feasible
        yield obj
        yield sol

        # provide c, A, b_ub, and ub
        model = Model()
        c = model.integer(2, lower_bound=-10, upper_bound=10)
        A = model.integer((3, 2), lower_bound=-10, upper_bound=10)
        b_ub = model.integer(3, lower_bound=-10, upper_bound=10)
        ub = model.integer(2, lower_bound=0, upper_bound=1)
        res = dwave.optimization.linprog(c, A=A, b_ub=b_ub, ub=ub)
        feasible = res.success
        obj = res.fun
        sol = res.x
        model.lock()

        yield res.lp
        yield feasible
        yield obj
        yield sol

        # provide all arguments
        model = Model()
        c = model.integer(2, lower_bound=-10, upper_bound=10)
        b_lb = model.integer(3, lower_bound=-10, upper_bound=10)
        A = model.integer((3, 2), lower_bound=-10, upper_bound=10)
        b_ub = model.integer(3, lower_bound=-10, upper_bound=10)
        A_eq = model.integer((4, 2), lower_bound=-10, upper_bound=10)
        b_eq = model.integer(4, lower_bound=-10, upper_bound=10)
        lb = model.integer(2, lower_bound=0, upper_bound=1)
        ub = model.integer(2, lower_bound=0, upper_bound=1)
        res = dwave.optimization.linprog(
            c, b_lb=b_lb, A=A, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub)
        feasible = res.success
        obj = res.fun
        sol = res.x
        model.lock()

        yield res.lp
        yield feasible
        yield obj
        yield sol

    def test_inputs_valid(self):
        from dwave.optimization.symbols import (
            LinearProgram,
            LinearProgramFeasible,
            LinearProgramObjectiveValue,
            LinearProgramSolution,
        )

        for name, kwargs in utils.iter_valid_lp_kwargs():
            with self.subTest(name):
                lp = LinearProgram(**kwargs)
                feas = LinearProgramFeasible(lp)
                res = LinearProgramObjectiveValue(lp)
                sol = LinearProgramSolution(lp)

                # smoke test that we can access the state in various ways without errors
                lp.model.states.resize(1)
                with lp.model.lock():
                    lp.state()
                    feas.state()
                    res.state()
                    sol.state()

    def test_inputs_invalid(self):
        for name, kwargs, msg in utils.iter_invalid_lp_kwargs():
            with self.subTest(name), self.assertRaisesRegex(ValueError, msg):
                dwave.optimization.symbols.LinearProgram(**kwargs)

    def test_unconstrained(self):
        model = Model()
        model.states.resize(1)
        c = model.integer(2, lower_bound=-10, upper_bound=10)
        res = dwave.optimization.linprog(c)
        sol = res.x
        with model.lock():
            c.set_state(0, [5, 5])
            np.testing.assert_array_equal(sol.state(), [0, 0])

    def test_two_variable_two_row(self):
        # min: -x0 + 4x1
        # such that:
        #      -3x0 + x1 <= 6
        #      -x0 - 2x1 >= -4
        #      x1 >= -3

        model = Model()
        model.states.resize(1)

        c = model.integer(2, lower_bound=-10)
        A_ub = model.integer((2, 2), lower_bound=-10)
        b_ub = model.integer(2, lower_bound=-10)
        lb = model.constant([-1e30, -3])

        res = dwave.optimization.linprog(c, A=A_ub, b_ub=b_ub, lb=lb)

        feasible = res.success
        obj = res.fun
        sol = res.x

        with model.lock():
            c.set_state(0, [-1, 4])
            A_ub.set_state(0, [[-3, 1], [1, 2]])
            b_ub.set_state(0, [6, 4])

            np.testing.assert_allclose(sol.state(), [10, -3])
            np.testing.assert_allclose(feasible.state(), 1)
            np.testing.assert_allclose(obj.state(), -1 * 10 + 4 * -3)

    def test_set_state(self):
        # min:
        #   -x0 - x1
        # such that:
        #   x0 + x1 <= 1
        #       -x0 <= 0
        #       -x1 <= 0
        model = Model()
        model.states.resize(1)

        c = model.constant([-1, -1])
        A = model.constant([[1, 1], [0, -1], [-1, 0]])
        b = model.constant([1, 0, 0])
        res = dwave.optimization.linprog(c, A=A, b_ub=b)

        feas = res.success
        lp = res.lp

        model.lock()

        lp._set_state(0, [0, 1])
        np.testing.assert_array_equal(lp.state(), [0, 1])
        self.assertEqual(feas.state(), True)

        lp._set_state(0, [1, 0])
        np.testing.assert_array_equal(lp.state(), [1, 0])
        self.assertEqual(feas.state(), True)

        lp._set_state(0, [1, 1])
        np.testing.assert_array_equal(lp.state(), [1, 1])
        self.assertEqual(feas.state(), False)

    def test_serialization_with_states(self):
        # min:
        #   -x0 - x1
        # such that:
        #   x0 + x1 <= 1
        #       -x0 <= 0
        #       -x1 <= 0
        model = Model()

        c = model.constant([-1, -1])
        A = model.constant([[1, 1], [0, -1], [-1, 0]])
        b = model.constant([1, 0, 0])
        res = dwave.optimization.linprog(c, A=A, b_ub=b)

        lp = res.lp

        model.states.resize(4)
        model.lock()

        lp._set_state(0, [0, 1])
        lp._set_state(1, [1, 0])
        # no states for 2
        lp._set_state(3, [1, 1])

        with self.subTest("model; lock=False"):
            with model.to_file(max_num_states=float("inf")) as f:
                copy = Model.from_file(f)  # lock=False by default

            _, _, _, lp_copy = copy.iter_symbols()

            self.assertFalse(copy.is_locked())
            with copy.lock():
                # these are all freshly calculated
                self.assertEqual(lp_copy.state(0).sum(), 1)
                self.assertEqual(lp_copy.state(1).sum(), 1)
                self.assertEqual(lp_copy.state(2).sum(), 1)
                self.assertEqual(lp_copy.state(3).sum(), 1)

        with self.subTest("model; lock=True"):
            with model.to_file(max_num_states=float("inf")) as f:
                copy = Model.from_file(f, lock=True)

            _, _, _, lp_copy = copy.iter_symbols()

            self.assertTrue(copy.is_locked())
            np.testing.assert_array_equal(lp_copy.state(0), [0, 1])
            np.testing.assert_array_equal(lp_copy.state(1), [1, 0])
            self.assertFalse(lp_copy.has_state(2))
            np.testing.assert_array_equal(lp_copy.state(3), [1, 1])

        with self.subTest("states"):
            with model.states.to_file() as f:
                model.states.clear()
                model.states.from_file(f)

            np.testing.assert_array_equal(lp.state(0), [0, 1])
            np.testing.assert_array_equal(lp.state(1), [1, 0])
            self.assertFalse(lp.has_state(2))
            np.testing.assert_array_equal(lp.state(3), [1, 1])


class TestMax(utils.ReduceTests):
    empty_requires_initial = True

    def op(self, x, *args, **kwargs):
        return x.max(*args, **kwargs)

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
        c = B.max(initial=6)
        d = B.max(initial=105)
        model.states.resize(1)
        model.lock()

        self.assertEqual(a.state(0), 4)
        self.assertEqual(b.state(0), 9)
        self.assertEqual(c.state(0), 9)
        self.assertEqual(d.state(0), 105)


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


class TestMean(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        c = model.constant([2, 3, 5, 1])
        mean = dwave.optimization.symbols.Mean(c)

        with model.lock():
            yield mean

    def test(self):
        from dwave.optimization.symbols import Mean
        model = Model()
        c = model.constant([2, 3, 5, 1])
        mean = dwave.optimization.mean(c)

        self.assertIsInstance(mean, Mean)

    def test_state(self):
        model = Model()
        c = model.constant([2, 3, 5, 1])
        mean = dwave.optimization.symbols.Mean(c)
        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(mean.state(0), np.array([2.75]))


class TestMin(utils.ReduceTests):
    empty_requires_initial = True

    def op(self, x, *args, **kwargs):
        return x.min(*args, **kwargs)

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
        c = B.min(initial=6)
        d = B.min(initial=-105)
        model.states.resize(1)
        model.lock()

        self.assertEqual(a.state(0), 0)
        self.assertEqual(b.state(0), 5)
        self.assertEqual(c.state(0), 5)
        self.assertEqual(d.state(0), -105)


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
        binary = x
        a = model.constant(5)
        b = model.constant(4)

        x += a  # type promotion to a NaryAdd
        self.assertIsInstance(x, dwave.optimization.symbols.NaryAdd)

        y = x
        x += b
        self.assertIs(x, y)  # subsequent should be in-place

        x += 5
        self.assertIs(x, y)  # subsequent should be in-place

        # Now we test adding an indexing node that depends on the output range
        i = model.constant(np.arange(20))[x]
        x += 1000000000
        self.assertIsNot(x, y)

        with model.lock():
            model.states.resize(1)
            binary.set_state(0, 0)
            self.assertEqual(x.state(), 1000000000 + 14)
            self.assertEqual(i.state(), 14)

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
        binary = x
        a = model.constant(5)
        b = model.constant(4)

        x *= a  # type promotion to a NaryAdd
        self.assertIsInstance(x, dwave.optimization.symbols.NaryMultiply)

        y = x
        x *= b
        self.assertIs(x, y)  # subsequent should be in-place

        x *= 5
        self.assertIs(x, y)  # subsequent should be in-place

        # Now we test adding an indexing node that depends on the output range
        i = model.constant(np.arange(101))[x]
        x *= 10
        self.assertIsNot(x, y)

        with model.lock():
            model.states.resize(1)
            binary.set_state(0, 1)
            self.assertEqual(x.state(), 1000)
            self.assertEqual(i.state(), 100)

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
        b = model.constant(2)
        c = model.constant(0)
        ab = logical_or(a, b)
        ac = logical_or(a, c)
        cc = logical_or(c, c)
        cb = logical_or(c, b)
        self.assertEqual(model.num_symbols(), 7)

        model.lock()
        model.states.resize(1)
        self.assertEqual(ab.state(0), 1)
        self.assertEqual(ac.state(0), 1)
        self.assertEqual(cc.state(0), 0)
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
        C = model.constant(np.arange(9).reshape(3, 3))
        a = A.prod(axis=0)
        b = B.prod(axis=1)
        c = C.prod(axis=1, initial=3)
        model.lock()
        yield a
        yield b
        yield c

    def test_initial(self):
        model = Model()
        model.states.resize(1)
        
        A = model.constant(np.arange(8).reshape((2, 2, 2)))

        with self.subTest(initial="howdy"):
            with self.assertRaises(TypeError):
                A.prod(axis=1, initial="howdy")
            self.assertEqual(model.num_symbols(), 1)  # no side-effects

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(8).reshape((2, 2, 2)))
        B = model.constant(np.arange(25).reshape((5, 5)))
        C = model.constant(np.arange(9).reshape(3, 3))
        a = A.prod(axis=0)
        b = B.prod(axis=1)
        c = C.prod(axis=1, initial=3)
        model.lock()
        model.states.resize(1)
        np.testing.assert_array_equal(a.state(0), np.prod(np.arange(8).reshape((2, 2, 2)), axis=0))
        np.testing.assert_array_equal(b.state(0), np.prod(np.arange(25).reshape((5, 5)), axis=1))
        np.testing.assert_array_equal(c.state(0), np.prod(np.arange(9).reshape(3, 3), axis=1, initial=3))

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
        C = model.constant(np.arange(9).reshape(3, 3))
        a = A.sum(axis=0)
        b = B.sum(axis=1)
        c = C.sum(axis=1, initial=3)
        model.lock()
        yield a
        yield b
        yield c

    def test_initial(self):
        model = Model()
        model.states.resize(1)
        
        A = model.constant(np.arange(8).reshape((2, 2, 2)))

        with self.subTest(initial="howdy"):
            with self.assertRaises(TypeError):
                A.sum(axis=1, initial="howdy")
            self.assertEqual(model.num_symbols(), 1)  # no side-effects

    def test_state(self):
        model = Model()

        A = model.constant(np.arange(8).reshape((2, 2, 2)))
        B = model.constant(np.arange(25).reshape((5, 5)))
        C = model.constant(np.arange(9).reshape(3, 3))
        a = A.sum(axis=0)
        b = B.sum(axis=1)
        c = C.sum(axis=1, initial=3)
        model.lock()
        model.states.resize(1)
        np.testing.assert_array_equal(a.state(0), np.sum(np.arange(8).reshape((2, 2, 2)), axis=0))
        np.testing.assert_array_equal(b.state(0), np.sum(np.arange(25).reshape((5, 5)), axis=1))
        np.testing.assert_array_equal(c.state(0), np.sum(np.arange(9).reshape(3, 3), axis=1, initial=3))

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


class TestProd(utils.ReduceTests):
    empty_requires_initial = False

    def op(self, x, *args, **kwargs):
        return x.prod(*args, **kwargs)

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

        B = model.constant(np.arange(12).reshape(3, 4))[model.set(3), :]
        syms.extend([B.reshape(-1), B.reshape(-1, 2, 2)])

        model.lock()
        yield from syms

    def test_dynamic(self):
        model = Model()
        s = model.set(10)
        r = s.reshape(-1, 1)
        self.assertEqual(r.shape(), (-1, 1))

        model.states.resize(1)
        with model.lock():
            s.set_state(0, [0, 1, 2])
            np.testing.assert_array_equal(r.state(), [[0], [1], [2]])

    def test_identity(self):
        model = Model()
        x = model.constant(np.arange(15).reshape(3, 5))
        self.assertEqual(x.id(), x.reshape(3, 5).id())

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


class TestResize(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        s = model.set(10)
        s_2x2 = s.resize((2, 2))
        c = model.constant(range(6))
        c_3 = dwave.optimization.mathematical.resize(c, 3)
        c_3x2 = c.resize((3, 2))
        with model.lock():
            yield from [s_2x2, c_3, c_3x2]

    def test_set(self):
        model = Model()
        model.states.resize(1)

        s = model.set(10)
        s_2x2 = s.resize((2, 2))
        s_5 = s.resize(5, fill_value=-1)

        with model.lock():
            s.set_state(0, [0, 1, 2])
            np.testing.assert_array_equal(s_2x2.state(), [[0, 1], [2, 0]])
            np.testing.assert_array_equal(s_5.state(), [0, 1, 2, -1, -1])

            s.set_state(0, list(reversed(range(10))))
            np.testing.assert_array_equal(s_2x2.state(), [[9, 8], [7, 6]])
            np.testing.assert_array_equal(s_5.state(), [9, 8, 7, 6, 5])


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


class TestSafeDivide(utils.BinaryOpTests):
    def generate_symbols(self):
        model = Model()

        a = model.constant(1)
        b = model.constant(2)
        zero = model.constant(0)

        w = safe_divide(a, b)
        x = safe_divide(b, a)
        y = safe_divide(a, zero)
        z = safe_divide(zero, a)
        with model.lock():
            yield from (w, x, y, z)

    def op(self, lhs, rhs):
        return np.divide(lhs, rhs, where=(rhs != 0))

    def symbol_op(self, lhs, rhs):
        return safe_divide(lhs, rhs)

    def test(self):
        model = Model()
        a = model.constant([-1, 0, 1, 2])
        b = model.constant([2, 1, 0, -1])
        x = safe_divide(a, b)

        model.states.resize(1)
        with model.lock():
            np.testing.assert_array_equal(x.state(), [-.5, 0, 0, -2])


class TestSin(utils.UnaryOpTests):
    def op(self, x):
        return np.sin(x)

    def symbol_op(self, x):
        return dwave.optimization.sin(x)


class TestSoftMax(utils.SymbolTests):
    def generate_symbols(self):
        model = Model()
        c = model.constant([-1.9, -2, 1.7, 1.6])
        sm = dwave.optimization.symbols.SoftMax(c)

        with model.lock():
            yield sm

    def test(self):
        from dwave.optimization.symbols import SoftMax
        model = Model()
        c = model.constant([-1.9, -2, 1.7, 1.6])
        sm = dwave.optimization.softmax(c)

        self.assertIsInstance(sm, SoftMax)

    def test_state(self):
        model = Model()
        c = model.constant([-1.9, -2, 1.7, 1.6])
        sm = dwave.optimization.symbols.SoftMax(c)
        model.states.resize(1)
        with model.lock():
            expected = np.array([0.0139628680773, 0.012634125499,  
                                 0.5110163194015, 0.4623866870215])
            np.testing.assert_array_almost_equal(sm.state(0), expected)


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
            with self.assertRaisesRegex(ValueError, r"^values must be a subset of range\(5\)$"):
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


class TestSum(utils.ReduceTests):
    empty_requires_initial = False

    def op(self, x, *args, **kwargs):
        return x.sum(*args, **kwargs)

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
        b = model.constant(2)
        c = model.constant(0)
        ab = logical_xor(a, b)
        ac = logical_xor(a, c)
        cc = logical_xor(c, c)
        cb = logical_xor(c, b)
        self.assertEqual(model.num_symbols(), 7)

        model.lock()
        model.states.resize(1)
        self.assertEqual(ab.state(0), 0)
        self.assertEqual(ac.state(0), 1)
        self.assertEqual(cc.state(0), 0)
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
