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

import unittest

import dwave.optimization.symbols

from dwave.optimization import Model, expression


class TestExpression(unittest.TestCase):
    def test_decorator(self):
        @expression
        def func(a, b, c):
            return (a + b) * c
        self.assertEqual(func._model.num_inputs(), 3)
        self.assertIsInstance(func._model.objective, dwave.optimization.symbols.Multiply)

    def test_decorator_with_bounds(self):
        @expression(a=dict(lower_bound=0), b=dict(integral=True))
        def func(a, b, c):
            return (a + b) * c

    def test_exceptions(self):
        with self.assertRaisesRegex(ValueError, "function must accept at least one argument"):
            expression(lambda: 1)
        with self.assertRaisesRegex(TypeError, "100 is not a callable object"):
            expression(100)
        with self.assertRaisesRegex(
                TypeError,
                r"expression\(\) takes 0 or 1 positional arguments but 3 were given",
        ):
            expression(lambda a: a, lambda b: b, lambda c: c)

    def test_lambda(self):
        identity = expression(lambda a: a)
        self.assertEqual(identity._model.num_inputs(), 1)
        self.assertIsInstance(identity._model.objective, dwave.optimization.symbols.Input)

        add = expression(lambda a, b: a + b)
        self.assertEqual(add._model.num_inputs(), 2)
        self.assertIsInstance(add._model.objective, dwave.optimization.symbols.Add)

    def test_lambda_with_bounds(self):
        identity = expression(lambda a: a, a=dict(lower_bound=0, upper_bound=10))
        a, = identity._model.iter_inputs()
        self.assertEqual(a.lower_bound(), 0)
        self.assertEqual(a.upper_bound(), 10)
        self.assertEqual(a.integral(), False)  # default
