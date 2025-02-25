# Copyright 2025 D-Wave
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
import os.path
import tempfile
import unittest

import numpy as np

from dwave.optimization import Model


class TestModelSerialization(unittest.TestCase):
    def test(self):
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

    def test_by_filename(self):
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

    def test_invalid_version_from_file(self):
        from dwave.optimization._model import DEFAULT_SERIALIZATION_VERSION

        model = Model()
        with io.BytesIO() as f:
            model.into_file(f)

            # check that the version is in the right place in the header and edit it
            f.seek(4)
            self.assertEqual(tuple(f.read(2)), DEFAULT_SERIALIZATION_VERSION)

            # now overwrite it and check that we fail
            f.seek(4)
            f.write(b"\xff\xff")

            f.seek(0)
            # the actual message is longer, but this is sufficient to test it's the
            # one being thrown
            with self.assertRaisesRegex(ValueError, "Unknown serialization format"):
                Model.from_file(f)

    def test_invalid_version_into_file(self):
        model = Model()
        with io.BytesIO() as f:
            with self.assertRaisesRegex(ValueError, "Unknown serialization format"):
                model.into_file(f, version=(255, 255))

        with self.assertRaisesRegex(ValueError, "Unknown serialization format"):
            with model.to_file(version=(255, 255)) as f:
                pass

    def test_max_num_states(self):
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

    def test_only_decision(self):
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

    def test_partially_initialized_states(self):
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


class TestStatesSerialization(unittest.TestCase):
    def test(self):
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

    def test_bad(self):
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
