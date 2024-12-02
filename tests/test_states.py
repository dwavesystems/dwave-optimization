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
import threading
import unittest

import numpy as np

import dwave.optimization

from dwave.optimization import Model


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
        self.assertIsInstance(model.states, dwave.optimization.states.States)

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
