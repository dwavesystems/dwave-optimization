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

import io
import operator
import unittest

import numpy as np

import dwave.optimization.symbols
from dwave.optimization import Model


class TestArraySymbol(unittest.TestCase):
    def test_abstract(self):
        from dwave.optimization.model import ArraySymbol
        with self.assertRaisesRegex(ValueError, "ArraySymbols cannot be constructed directly"):
            ArraySymbol()

    def test_bool(self):
        from dwave.optimization.model import ArraySymbol

        # bypass the init, this should be done very carefully as it can lead to
        # segfaults dependig on what methods are accessed!
        symbol = ArraySymbol.__new__(ArraySymbol)

        with self.assertRaises(ValueError):
            bool(symbol)

    def test_combined_indexing_with_integers(self):
        model = Model()
        a = model.constant(np.arange(27).reshape(3, 3, 3))
        x = model.list(3)

        y = a[x, x, 2]

        model.lock()
        self.assertEqual(model.num_symbols(), 4)  # an intermediate node was created

        model.states.resize(1)

        xarr = np.asarray(x.state(), dtype=int)
        aarr = np.asarray(a)

        np.testing.assert_array_equal(y.state(), aarr[xarr, xarr, 2])

    def test_combined_indexing_with_slices(self):
        model = Model()
        a = model.constant(np.arange(27).reshape(3, 3, 3))
        x = model.list(3)

        y = a[x, x, 1:3]

        model.lock()
        self.assertEqual(model.num_symbols(), 4)  # an intermediate node was created

        model.states.resize(1)

        xarr = np.asarray(x.state(), dtype=int)
        aarr = np.asarray(a)

        np.testing.assert_array_equal(y.state(), aarr[xarr, xarr, 1:3])

    def test_operator_types(self):
        # For each, test that we get the right class from the operator and that
        # incorrect types returns NotImplemented
        # The actual correctness of the resulting symbol is tested elsewhere

        model = Model()
        x = model.binary()
        y = model.binary()

        class UnknownType():
            pass

        operators = [
            (operator.add, "__add__", dwave.optimization.symbols.Add),
            (operator.eq, "__eq__", dwave.optimization.symbols.Equal),
            (operator.iadd, "__iadd__", dwave.optimization.symbols.NaryAdd),
            (operator.imul, "__imul__", dwave.optimization.symbols.NaryMultiply),
            (operator.le, "__le__", dwave.optimization.symbols.LessEqual),
            (operator.mul, "__mul__", dwave.optimization.symbols.Multiply),
            (operator.sub, "__sub__", dwave.optimization.symbols.Subtract),
        ]

        for op, method, cls in operators:
            with self.subTest(method):
                self.assertIsInstance(op(x, y), cls)
                self.assertIs(getattr(x, method)(UnknownType()), NotImplemented)

        # The operators that don't fit as neatly into the above

        with self.subTest("__pow__"):
            self.assertIsInstance(x ** 1, type(x))
            self.assertIsInstance(x ** 1, dwave.optimization.model.ArraySymbol)
            self.assertIsInstance(x ** 2, dwave.optimization.symbols.Square)
            self.assertIsInstance(x ** 3, dwave.optimization.symbols.NaryMultiply)
            self.assertIsInstance(x ** 4, dwave.optimization.symbols.NaryMultiply)
            self.assertIsInstance(x ** 5, dwave.optimization.symbols.NaryMultiply)
            self.assertIs(x.__pow__(UnknownType()), NotImplemented)

    class IndexTester:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, index):
            if not isinstance(index, tuple):
                return self[(index,)]
            i0, i1 = dwave.optimization._model._split_indices(index)
            np.testing.assert_array_equal(self.array[index], self.array[i0][i1])

    def test_split_indices(self):
        a1d = np.arange(5)
        a2d = np.arange(5 * 6).reshape(5, 6)
        a4d = np.arange(5 * 6 * 7 * 8).reshape(5, 6, 7, 8)

        for arr in [a1d, a2d,  a4d]:
            test = self.IndexTester(arr)

            test[:]
            test[0]
            test[np.asarray([0, 1, 2])]
            test[1:]
            test[1:4:2]

            if arr.ndim < 2:
                continue

            test[:, :]
            test[0, 1]
            test[:2, :]
            test[:, 1::2]
            test[0, :]
            test[:, 3]
            test[::2, 2]
            test[np.asarray([0, 2, 1]), np.asarray([0, 0, 0])]
            test[np.asarray([0, 2, 1]), :]
            test[np.asarray([0, 2, 1]), 3]
            test[:, np.asarray([0, 2, 1])]
            test[3, np.asarray([0, 2, 1])]

            if arr.ndim < 3:
                continue

            test[:, :, :, :]
            test[0, 1, 2, 4]
            test[::2, 0, ::2, :]

            # two different types of combined indexing
            test[np.asarray([0, 2, 1]), :, np.asarray([0, 0, 0]), np.asarray([0, 0, 0])]
            test[np.asarray([0, 2, 1]), np.asarray([0, 0, 0]), np.asarray([0, 0, 0]), :]


class TestModel(unittest.TestCase):
    def test(self):
        model = Model()

    def test_add_constraint(self):
        model = Model()
        x = model.binary(5)

        with self.assertRaisesRegex(ValueError,
                                    "The truth value of an array with "
                                    "more than one element is ambiguous"):
            model.add_constraint(x)

        c_direct = model.add_constraint(x.all())

        model.states.resize(1)
        model.lock()

        c_iter, = model.iter_constraints()

        self.assertEqual(x.state(0).all(), c_iter.state(0))
        self.assertEqual(x.state(0).all(), c_direct.state(0))

        x.set_state(0, [0, 0, 0, 0, 0])
        self.assertEqual(x.state(0).all(), c_iter.state(0))
        self.assertEqual(x.state(0).all(), c_direct.state(0))

        x.set_state(0, [0, 0, 0, 0, 0])
        self.assertFalse(c_iter.state(0))
        self.assertFalse(c_direct.state(0))

        x.set_state(0, [1, 1, 1, 1, 1])
        self.assertTrue(c_iter.state(0))
        self.assertTrue(c_direct.state(0))

    def test_feasible(self):
        model = Model()
        i = model.integer()
        c = model.constant(5)
        model.add_constraint(i <= c)

        with model.lock():
            # Check expected exception if no states initialize
            with self.assertRaises(ValueError):
                model.feasible(0)

            model.states.resize(1)
            i.set_state(0, 1)

            # Check that True is returned for feasible state
            self.assertTrue(model.feasible(0))

            # Check expected exception for index out of range with initialized state
            with self.assertRaises(ValueError):
                model.feasible(1)

            i.set_state(0, 6)

            # Check that False is returned for infeasible state
            self.assertFalse(model.feasible(0))

    def test_lock(self):
        model = Model()

        # Works as a context manager
        with model.lock():
            self.assertTrue(model.is_locked())
        self.assertFalse(model.is_locked())

        # Works in case of an exception
        with model.lock(), self.assertRaises(ValueError):
            self.assertTrue(model.is_locked())
            raise ValueError
        self.assertFalse(model.is_locked())

        # Works as a reentrant context manager
        with model.lock():
            self.assertTrue(model.is_locked())
            with model.lock():
                self.assertTrue(model.is_locked())
            self.assertTrue(model.is_locked())
        self.assertFalse(model.is_locked())

        # Can do weird but valid things with explicit and context locking
        with model.lock():
            self.assertTrue(model.is_locked())
            model.unlock()  # rude, but allowed
            self.assertFalse(model.is_locked())
        self.assertFalse(model.is_locked())
        with model.lock():
            self.assertTrue(model.is_locked())
            model.lock()  # rude, but allowed
            self.assertTrue(model.is_locked())
        self.assertTrue(model.is_locked())

    def test_minimize(self):
        model = Model()

        model.states.resize(1)

        model.minimize(model.constant(5))
        self.assertEqual(model.objective.state(), 5)

        model.minimize(model.constant([6]))
        self.assertEqual(model.objective.state(), 6)

        model.minimize(model.constant([[7]]))
        self.assertEqual(model.objective.state(), 7)

    def test_remove_unused_symbols(self):
        with self.subTest("all unused"):
            model = Model()

            # add three symbols to the model without keeping them in the namespace
            # or adding to constraints/objective
            model.constant(0) + model.integer()

            num_removed = model.remove_unused_symbols()

            # only the decision is kept
            self.assertEqual(num_removed, 2)
            self.assertEqual(model.num_symbols(), 1)
            x, = model.iter_symbols()
            self.assertIsInstance(x, dwave.optimization.symbols.IntegerVariable)

        with self.subTest("all used in objective"):
            model = Model()

            # add three symbols to the model without keeping them in the namespace
            # or adding to constraints/objective
            model.minimize(model.constant(0) + model.integer())

            num_removed = model.remove_unused_symbols()

            # everything is kept
            self.assertEqual(num_removed, 0)
            self.assertEqual(model.num_symbols(), 3)

        with self.subTest("all kept in the namespace"):
            model = Model()

            # add three symbols to the model and keep them in the namespace
            y = model.constant(0) + model.integer()

            num_removed = model.remove_unused_symbols()

            # everything is kept
            self.assertEqual(num_removed, 0)
            self.assertEqual(model.num_symbols(), 3)

            # now delete the namespace symbol
            del y

            num_removed = model.remove_unused_symbols()

            # only the decision is kept
            self.assertEqual(num_removed, 2)
            self.assertEqual(model.num_symbols(), 1)

        with self.subTest("disjoint lists"):
            model = Model()

            base, lists = model.disjoint_lists(10, 4)

            # only use some of the lists
            model.minimize(lists[0].sum())
            model.add_constraint(lists[1].sum() <= model.constant(3))

            lists[2].prod()  # this one will hopefully be removed

            self.assertEqual(model.num_symbols(), 10)

            # make sure they aren't being kept alive by other objects
            del lists
            del base

            num_removed = model.remove_unused_symbols()

            # only 1 is removed
            self.assertEqual(num_removed, 1)
            self.assertEqual(model.num_symbols(), 9)

    def test_state_size(self):
        self.assertEqual(Model().state_size(), 0)

        # one constant
        model = Model()
        model.constant(np.arange(25).reshape(5, 5))
        self.assertEqual(model.state_size(), 25 * 8)

    def test_to_networkx(self):
        try:
            import networkx as nx
        except ImportError:
            return self.skipTest("NetworkX is not installed")

        model = Model()
        a = model.binary()
        b = model.binary()
        ab = a * b

        G = model.to_networkx()

        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 2)

        # the repr as labels is an implementation detail and subject to change
        self.assertIn(repr(a), G.nodes)
        self.assertIn(repr(b), G.nodes)
        self.assertIn(repr(ab), G.nodes)
        self.assertIn((repr(a), repr(ab)), G.edges)
        self.assertIn((repr(b), repr(ab)), G.edges)

        # graph created is deterministic
        self.assertTrue(nx.utils.graphs_equal(G, model.to_networkx()))

    def test_to_networkx_multigraph(self):
        try:
            import networkx as nx
        except ImportError:
            return self.skipTest("NetworkX is not installed")

        model = Model()
        a = model.binary()
        aa = a * a  # two edges to the same node

        G = model.to_networkx()

        # the repr as labels is an implementation detail and subject to change
        self.assertEqual(len(G.nodes), 2)
        self.assertEqual(len(G.edges), 2)
        self.assertIn(repr(a), G.nodes)
        self.assertIn(repr(aa), G.nodes)
        self.assertIn((repr(a), repr(aa)), G.edges)

        # graph created is deterministic
        self.assertTrue(nx.utils.graphs_equal(G, model.to_networkx()))

    def test_to_networkx_objective_and_constraints(self):
        try:
            import networkx as nx
        except ImportError:
            return self.skipTest("NetworkX is not installed")

        model = Model()
        a = model.binary()
        b = model.binary()
        model.minimize(a * b)

        G = model.to_networkx()
        self.assertEqual(len(G.nodes), 4)  # 3 symbols + "minimize"
        self.assertEqual(len(G.edges), 3)
        self.assertIn("minimize", G.nodes)

        model.add_constraint(a <= b)
        G = model.to_networkx()
        self.assertEqual(len(G.nodes), 6)
        self.assertEqual(len(G.edges), 6)
        self.assertIn("constraint(s)", G.nodes)


class TestSymbol(unittest.TestCase):
    def test_abstract(self):
        from dwave.optimization.model import Symbol
        with self.assertRaisesRegex(ValueError, "Symbols cannot be constructed directly"):
            Symbol()

    def test_id(self):
        model = Model()
        c0 = model.constant(5)
        c1, = model.iter_symbols()
        c2 = model.constant(5)

        self.assertIsInstance(c0.id(), int)
        self.assertEqual(c0.id(), c1.id())
        self.assertNotEqual(c0.id(), c2.id())

    def test_repr(self):
        model = Model()
        c0 = model.constant(5)
        c1, = model.iter_symbols()
        c2 = model.constant(5)

        # the specific form is an implementation detail, but different symbols
        # representing the same underlying node should have the same repr
        self.assertEqual(repr(c0), repr(c1))
        self.assertNotEqual(repr(c0), repr(c2))
