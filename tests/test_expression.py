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

from dwave.optimization import Model, expression
from dwave.optimization.symbols import NaryReduce


class TestExpression(unittest.TestCase):
    def test_lambda(self):
        identity = expression(lambda a: a)
        print(identity)

    def test_named(self):

        @expression
        def func(a, b, c):
            return (a + b) * c

        NaryReduce(func)
