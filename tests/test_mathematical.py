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

# Note: Many of the mathematical.py functions are thin wrappers around symbols
# and therefore are tested in tests/test_symbols.py. This file is for testing
# the other methods.

import unittest

import numpy as np

import dwave.optimization

from dwave.optimization import Model


class TestStack(unittest.TestCase):
    def test_scalars(self):
        model = Model()
        symbols = [model.constant(1), model.constant(2), model.constant(3)]
        s = dwave.optimization.stack(symbols)
        with model.lock():
            model.states.resize(1)
            self.assertIsInstance(s, dwave.optimization.model.ArraySymbol)
            self.assertEqual(s.shape(), (3, ))
            self.assertEqual(s.ndim(), 1)
            np.testing.assert_array_equal(s.state(0), np.arange(1, 4))

    def test_1d_arrays(self):
        model = Model()
        A = model.constant(np.arange(3).reshape((3, )))
        B = model.constant(np.arange(3, 6).reshape((3, )))
        s0 = dwave.optimization.stack((A, B), axis=0)
        s1 = dwave.optimization.stack((A, B), axis=1)
        with model.lock():
            model.states.resize(1)
            np.testing.assert_array_equal(
                s0.state(0), np.stack((A, B), axis=0))
            np.testing.assert_array_equal(
                s1.state(0), np.stack((A, B), axis=1))

    def test_2d_arrays(self):
        model = Model()
        A = model.constant(np.arange(4).reshape((2, 2)))
        B = model.constant(np.arange(4, 8).reshape((2, 2)))
        s0 = dwave.optimization.stack((A, B), axis=0)
        s1 = dwave.optimization.stack((A, B), axis=1)
        s2 = dwave.optimization.stack((A, B), axis=2)
        with model.lock():
            model.states.resize(1)
            np.testing.assert_array_equal(
                s0.state(0), np.stack((A, B), axis=0))
            np.testing.assert_array_equal(
                s1.state(0), np.stack((A, B), axis=1))
            np.testing.assert_array_equal(
                s2.state(0), np.stack((A, B), axis=2))

    def test_nd_arrays(self):
        rng = np.random.default_rng(1)
        dims = rng.integers(0, 5)
        shape = tuple(rng.integers(1, 4) for _ in range(dims + 1))
        model = Model()
        A = np.random.randint(0, 10, shape)
        B = np.random.randint(0, 10, shape)
        C = np.random.randint(0, 10, shape)
        symbols = [model.constant(A), model.constant(B), model.constant(C)]
        for axis in range(dims+1):
            s = dwave.optimization.stack(symbols, axis)
            with model.lock():
                model.states.resize(1)
                np.testing.assert_array_equal(
                    s.state(0),
                    np.stack((A, B, C), axis))

    def test_errors(self):
        with self.subTest("axis out of bounds"):
            model = Model()
            A = model.constant(np.arange(9).reshape((3, 1, 3)))
            B = model.constant(np.arange(9).reshape((3, 1, 3)))
            with self.assertRaisesRegex(
                ValueError,
                (r"axis 4 is out of bounds for array of dimension 4")
            ):
                dwave.optimization.stack((A, B), axis=4)

        with self.subTest("input arrays must have same shape"):
            model = Model()
            A = model.constant(np.arange(4).reshape((2, 2)))
            B = model.constant(np.arange(8).reshape((4, 2)))
            with self.assertRaisesRegex(
                ValueError,
                (r"all input array symbols must have the same shape")
            ):
                dwave.optimization.stack((A, B))

        with self.subTest("input arrays must have same shape and ndim"):
            model = Model()
            A = model.constant(np.arange(4).reshape((2, 2)))
            B = model.constant(np.arange(4).reshape((2, 1, 2)))
            with self.assertRaisesRegex(
                ValueError,
                (r"all input array symbols must have the same shape")
            ):
                dwave.optimization.stack((A, B))

        with self.subTest("at least one input array is required"):
            model = Model()
            with self.assertRaisesRegex(
                ValueError,
                (r"need at least one array symbol to stack")
            ):
                dwave.optimization.stack([])

    def test_single_array_symbol(self):
        model = Model()
        A = model.constant(np.arange(4).reshape(2, 2))
        s = dwave.optimization.stack(A)
        self.assertEqual(s.shape(), (2, 2))
        self.assertEqual(s.ndim(), 2)
        self.assertTrue(s is A)
        self.assertIsInstance(s, dwave.optimization.symbols.Constant)

    def test_single_array(self):
        model = Model()
        A = model.constant(np.arange(4).reshape((2, 2)))
        s = dwave.optimization.stack((A, ))
        with model.lock():
            model.states.resize(1)
            self.assertEqual(s.shape(), (1, 2, 2))
            self.assertEqual(s.ndim(), 3)
            np.testing.assert_array_equal(s.state(0), np.stack((A,)))
