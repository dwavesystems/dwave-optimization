# Copyright 2024 D-Wave Inc.
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
from dwave.optimization.model import Expression


class TestExpression(unittest.TestCase):
    def test(self):
        Expression()

    def test_unsupported_symbols(self):
        # Can't add decisions to an Expression, even manually
        expr = Expression()
        with self.assertRaises(TypeError):
            dwave.optimization.symbols.IntegerVariable(expr)

        # Can't add other symbols, e.g. Reshape
        expr = Expression()
        inp = expr.input(0, 1, False)
        with self.assertRaises(TypeError):
            dwave.optimization.symbols.Reshape(inp, (1, 1, 1))

    def test_num_inputs(self):
        expr = Expression()
        self.assertEqual(expr.num_inputs(), 0)

        inp0 = expr.input(-1, 1, True)
        self.assertEqual(expr.num_inputs(), 1)

        inp1 = expr.input(-1, 1, True)
        self.assertEqual(expr.num_inputs(), 2)

        inp0 + inp1
        self.assertEqual(expr.num_inputs(), 2)
        self.assertEqual(expr.num_nodes(), 3)

        expr.input(-1, 1, True)
        self.assertEqual(expr.num_inputs(), 3)
        self.assertEqual(expr.num_nodes(), 4)

    def test_iter_inputs(self):
        expr = Expression()
        self.assertListEqual(list(expr.iter_inputs()), [])

        inp0 = expr.input(-1, 1, True)
        symbols = list(expr.iter_inputs())
        self.assertEqual(len(symbols), 1)
        self.assertTrue(inp0.equals(symbols[0]))

        inp1 = expr.input(-1, 1, True)
        symbols = list(expr.iter_inputs())
        self.assertEqual(len(symbols), 2)
        self.assertTrue(inp0.equals(symbols[0]))
        self.assertTrue(inp1.equals(symbols[1]))

        inp0 + inp1
        symbols = list(expr.iter_inputs())
        self.assertEqual(len(symbols), 2)

        inp2 = expr.input(-1, 1, True)
        symbols = list(expr.iter_inputs())
        self.assertEqual(len(symbols), 3)
        self.assertTrue(inp2.equals(symbols[2]))

    def test_constants(self):
        expr = Expression()
        c0, c1 = expr.constant(5), expr.constant(-7.5)
        c0 + c1
