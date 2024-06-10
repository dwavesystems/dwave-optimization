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

import concurrent.futures
import io
import os.path
import tempfile
import threading
import unittest

import numpy as np

import dwave.optimization.symbols
from dwave.optimization import Model


class TestArrayObserver(unittest.TestCase):
    class IndexTester:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, index):
            if not isinstance(index, tuple):
                return self[(index,)]
            i0, i1 = dwave.optimization.model._split_indices(index)
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

        model.add_constraint(x.all())

        model.states.resize(1)
        model.lock()

        c, = model.iter_constraints()

        self.assertEqual(x.state(0).all(), c.state(0))

        x.set_state(0, [0, 0, 0, 0, 0])
        self.assertEqual(x.state(0).all(), c.state(0))

        x.set_state(0, [0, 0, 0, 0, 0])
        self.assertFalse(c.state(0))

        x.set_state(0, [1, 1, 1, 1, 1])
        self.assertTrue(c.state(0))

    def test_lock(self):
        model = Model()

        # Works as a context manager
        with model.lock():
            self.assertTrue(model.is_locked())
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

    def test_serialization(self):
        # Create a simple model
        model = Model()
        x = model.list(5)
        W = model.constant(np.arange(25).reshape((5, 5)))
        model.minimize(W[x, :][:, x].sum())

        # Serialize, then deserialize the model
        with model.to_file() as f:
            new = Model.from_file(f)

        new.lock()
        new.states.resize(3)
        a, *_ = new.iter_symbols()
        a.set_state(0, [2, 4, 3, 1, 0])
        # don't set state 1
        a.state(2)  # default initialize

        # Attach the new states to the original model
        with new.states.to_file() as f:
            model.states.from_file(f)

        self.assertEqual(model.states.size(), 3)
        np.testing.assert_array_equal(x.state(0), [2, 4, 3, 1, 0])
        self.assertFalse(a.has_state(1))
        np.testing.assert_array_equal(a.state(2), range(5))


    def test_serialization_by_filename(self):
        model = Model()
        c = model.constant([0, 1, 2, 3, 4])
        x = model.list(5)
        model.minimize(c[x].sum())

        with tempfile.TemporaryDirectory() as dirname:
            fname = os.path.join(dirname, "temp.nl")

            model.into_file(fname)

            new = Model.from_file(fname)

        # todo: use a model equality test function once we have it
        for n0, n1 in zip(model.iter_symbols(), new.iter_symbols()):
            self.assertIs(type(n0), type(n1))

    def test_serialization_max_num_states(self):
        model = Model()
        model.states.resize(4)

        with self.subTest("large max_num_states"):
            with model.to_file(max_num_states=100) as f:
                new = Model.from_file(f)
            self.assertEqual(new.states.size(), 4)

        with self.subTest("small max_num_states"):
            with model.to_file(max_num_states=2) as f:
                new = Model.from_file(f)
            self.assertEqual(new.states.size(), 2)

    def test_serialization_only_decision(self):
        model = Model()
        a = model.list(5)
        model.constant(100)
        b = model.list(6)
        model.constant(14)

        with model.to_file(only_decision="truthy") as f:
            new = Model.from_file(f)

        self.assertEqual(new.num_nodes(), 2)
        x, y = new.iter_symbols()
        self.assertEqual(x.shape(), a.shape())
        self.assertEqual(y.shape(), b.shape())

    def test_serialization_partially_initialized_states(self):
        model = Model()
        x = model.list(10)
        model.states.resize(3)
        x.set_state(0, list(reversed(range(10))))
        # don't set state 1
        x.state(2)  # default initialize

        with model.to_file(max_num_states=model.states.size()) as f:
            new = Model.from_file(f)

        a, = new.iter_symbols()
        self.assertEqual(new.states.size(), 3)
        np.testing.assert_array_equal(a.state(0), x.state(0))
        self.assertFalse(a.has_state(1))
        np.testing.assert_array_equal(a.state(2), x.state(2))

    def test_state_size(self):
        self.assertEqual(Model().state_size(), 0)

        # one constant
        model = Model()
        model.constant(np.arange(25).reshape(5, 5))
        self.assertEqual(model.state_size(), 25 * 8)


class TestStates(unittest.TestCase):
    def test_clear(self):
        model = Model()
        model.states.resize(100)
        self.assertEqual(model.states.size(), 100)
        model.states.clear()
        self.assertEqual(model.states.size(), 0)

    def test_from_future(self):
        model = Model()
        x = model.list(10)
        model.minimize(x.sum())

        model.states.resize(5)

        fp = io.BytesIO()
        model.lock()
        model.states.into_file(fp)
        fp.seek(0)  # into file doesn't return to the beginning
        model.states.resize(0)

        future = concurrent.futures.Future()

        def hook(model, future):
            model.states.from_file(future.result())

        model.states.from_future(future, hook)

        future.set_result(fp)

        x.state(0)

    def test_resolve(self):
        model = Model()
        x = model.list(10)
        model.minimize(x.sum())

        model.states.resize(5)

        fp = io.BytesIO()
        model.lock()
        model.states.into_file(fp)
        fp.seek(0)  # into file doesn't return to the beginning
        model.states.resize(0)

        executor = concurrent.futures.ThreadPoolExecutor()
        future = concurrent.futures.Future()
        resolved = threading.Event()

        def hook(model, future):
            model.states.from_file(future.result())

        model.states.from_future(future, hook)

        def worker():
            x.state(0)
            resolved.set()

        executor.submit(worker)

        self.assertFalse(resolved.wait(timeout=0.1))

        future.set_result(fp)

        executor.shutdown(wait=True)

        self.assertTrue(resolved.is_set())

    def test_set_state_with_successors(self):
        model = Model()
        b = model.integer()
        c = model.constant(2)
        model.minimize(b + c)
        model.states.resize(1)

        b.set_state(0, 1) 
        self.assertEqual(b.state(0), 1)

        # can set state when locked
        with model.lock():
            b.set_state(0, 2)
            self.assertEqual(b.state(0), 2)

        b.set_state(0, 3)
        self.assertEqual(b.state(0), 3)

    def test_init(self):
        model = Model()

        self.assertTrue(hasattr(model, "states"))
        self.assertIsInstance(model.states, dwave.optimization.model.States)

        # by default there are no states
        self.assertEqual(len(model.states), 0)
        self.assertEqual(model.states.size(), 0)

    def test_resize(self):
        model = Model()

        # cannot resize to a negative number
        with self.assertRaises(ValueError):
            model.states.resize(-1)

        # we can resize if the model and states are unlocked
        model.states.resize(5)
        self.assertEqual(model.states.size(), 5)

        # we can resize the states if the model is locked and the states are
        # unlocked
        model.lock()
        model.states.resize(10)
        self.assertEqual(model.states.size(), 10)

    def test_serialization(self):
        model = Model()
        x = model.list(10)
        model.states.resize(3)
        x.set_state(0, list(reversed(range(10))))
        # don't set state 1
        x.state(2)  # default initialize

        # Get another model with the same shape. This won't work in general
        # unless you're very careful to always insert nodes in the same order
        new = Model()
        a = new.list(10)

        with model.states.to_file() as f:
            new.states.from_file(f)

        self.assertEqual(new.states.size(), 3)
        np.testing.assert_array_equal(a.state(0), x.state(0))
        self.assertFalse(a.has_state(1))
        np.testing.assert_array_equal(a.state(2), x.state(2))

    def test_serialization_bad(self):
        model = Model()
        x = model.list(10)
        model.states.resize(1)
        x.set_state(0, list(reversed(range(10))))

        with self.subTest("different node class"):
            new = Model()
            new.constant(10)

            with model.states.to_file() as f:
                with self.assertRaises(ValueError):
                    new.states.from_file(f)

        # todo: uncomment once we have proper node equality testing
        # with self.subTest("different node shape"):
        #     new = Model()
        #     new.list(9)

        #     with model.states.to_file() as f:
        #         with self.assertRaises(ValueError):
        #             new.states.from_file(f)
