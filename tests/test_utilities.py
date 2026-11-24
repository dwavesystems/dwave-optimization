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

import os
import pathlib
import tempfile
import typing
import unittest

import numpy as np

import dwave.optimization
import dwave.optimization.typing
import dwave.optimization.utilities


class Test_NoValue(unittest.TestCase):
    def test(self):
        self.assertIs(
            dwave.optimization.utilities._NoValue,
            dwave.optimization.utilities._NoValueType(),
        )


class Test_file_object_arg(unittest.TestCase):
    def test_file_like(self):
        @dwave.optimization.utilities._file_object_arg(arg_index=1, arg_name="file", mode="rb")
        def func(base: typing.BinaryIO, file: typing.BinaryIO, *_, **__):
            # it was a pass-through
            self.assertIs(base, file)

        with tempfile.TemporaryFile("rb") as f:
            func(f, f)
            func(f, file=f)
            func(f, f, 1)
            func(f, f, a=1)
            func(f, file=f, a=1)

    @unittest.skipUnless(os.name == "posix", "on POSIX the tempfile can always be opened again")
    def test_path_like(self):
        @dwave.optimization.utilities._file_object_arg(arg_index=1, arg_name="file", mode="rb")
        def str_func(base: typing.IO, file: str | os.PathLike[str] | typing.IO, *_, **__):
            self.assertEqual(base.name, file.name)
            self.assertTrue(file.readable())
            self.assertFalse(file.writable())

        with tempfile.NamedTemporaryFile("rb") as f:
            str_func(f, f.name)
            str_func(f, file=f.name)
            str_func(f, f.name, 1)
            str_func(f, f.name, a=1)
            str_func(f, file=f.name, a=1)

        @dwave.optimization.utilities._file_object_arg(arg_index=1, arg_name="file", mode="rb")
        def path_func(base: typing.IO, file: os.PathLike[str] | typing.IO, *_, **__):
            self.assertEqual(base.name, file.name)
            self.assertTrue(file.readable())
            self.assertFalse(file.writable())

        with tempfile.NamedTemporaryFile("rb") as f:
            pth = pathlib.Path(f.name)
            path_func(f, pth)
            path_func(f, file=pth)
            path_func(f, pth, 1)
            path_func(f, pth, a=1)
            path_func(f, file=pth, a=1)

        @dwave.optimization.utilities._file_object_arg(arg_index=1, arg_name="file", mode="wb")
        def path_func(base: typing.IO, file: os.PathLike[str] | typing.IO, *_, **__):
            self.assertEqual(base.name, file.name)
            self.assertFalse(file.readable())
            self.assertTrue(file.writable())

        with tempfile.NamedTemporaryFile("wb") as f:
            pth = pathlib.Path(f.name)
            path_func(f, pth)
            path_func(f, file=pth)
            path_func(f, pth, 1)
            path_func(f, pth, a=1)
            path_func(f, file=pth, a=1)


class Test_split_indices(unittest.TestCase):
    class IndexTester:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, index):
            if not isinstance(index, tuple):
                return self[(index,)]
            dims, basic, advanced = dwave.optimization.utilities._split_indices(
                self.array.shape, index
            )
            np.testing.assert_array_equal(
                self.array[index], np.expand_dims(self.array, dims)[basic][advanced]
            )

    def test_equivalence(self):
        a1d = np.arange(5)
        a2d = np.arange(5 * 6).reshape(5, 6)
        a4d = np.arange(5 * 6 * 7 * 8).reshape(5, 6, 7, 8)

        for arr in [a1d, a2d, a4d]:
            test = self.IndexTester(arr)

            test[:]
            test[0]
            test[np.int8(0)]
            test[np.asarray([0, 1, 2])]
            test[1:]
            test[1:4:2]
            test[:, np.newaxis]
            test[np.newaxis, :]
            test[np.newaxis, :, np.newaxis]
            test[...]
            test[..., 0]
            test[:, ...]
            test[np.newaxis, ...]
            test[..., np.newaxis]
            test[:, ..., np.newaxis]
            test[..., :, np.newaxis]

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
            test[..., 3, np.asarray([0, 2, 1])]
            test[np.newaxis, ..., 3, np.asarray([0, 2, 1])]

            if arr.ndim < 3:
                continue

            test[:, :, :, :]
            test[0, 1, 2, 4]
            test[::2, 0, ::2, :]
            test[:, ..., :]
            test[:, ..., ::2]
            test[np.newaxis, :]
            test[..., 0, np.newaxis, 2]

            # known issue: https://github.com/dwavesystems/dwave-optimization/issues/465
            # test[:, 0, :, [2]]

            test[np.asarray([0, 2, 1]), :, np.asarray([0, 0, 0]), np.asarray([0, 0, 0])]
            test[np.asarray([0, 2, 1]), np.asarray([0, 0, 0]), np.asarray([0, 0, 0]), :]
