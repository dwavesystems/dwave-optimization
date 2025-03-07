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

"""
This file is designed for backwards compatibility! So it tries to use
the most stable APIs.
Specifically, it needs to work with dwave-optimization 0.5.2, the last
version that only supported serialization version 0.1
"""

from __future__ import annotations

import os.path
import tempfile
import typing
import unittest

import numpy as np

import dwave.optimization
from dwave.optimization import Model


class TestSerialization(unittest.TestCase):
    """For each model to be tested we expect a ``make_<model>()`` method."""
    def assertModelEqual(self, lhs: Model, rhs: Model):
        """Assert two models are the same."""
        with lhs.lock(), rhs.lock():
            # The models must have the same number of symbols
            self.assertEqual(lhs.num_symbols(), rhs.num_symbols())

            # The models must have the same number of states
            self.assertEqual(lhs.states.size(), rhs.states.size())

            # All nodes in the model need to match, and to have the same states
            for ls, rs in zip(lhs.iter_symbols(), rhs.iter_symbols()):
                self.assertTrue(ls.maybe_equals(rs))

                # If the nodes have states, check that they are all equal
                # If all are maybe_equal then this amounts to a full equality
                # check for the model as a whole.
                if isinstance(ls, dwave.optimization.model.ArraySymbol):
                    for i in range(lhs.states.size()):
                        self.assertEqual(ls.has_state(i), rs.has_state(i))
                        if ls.has_state(i):
                            np.testing.assert_array_equal(ls.state(i), rs.state(i))

    @classmethod
    def iter_names(cls) -> typing.Iterator[str]:
        """Iterate over the model names."""
        for method_name in dir(cls):
            if method_name.startswith("make_"):
                yield method_name.removeprefix("make_")

    def load(self, directory: str):
        """Load & test all models in the given directory"""
        for name in self.iter_names():
            print("loading", name)  # could add a verbose flag to toggle

            base_model = getattr(self, "make_" + name)()

            # test model equality
            self.assertModelEqual(
                Model.from_file(os.path.join(directory, "model", name + ".nl")),
                base_model
                )

            # test states equality
            test_model = getattr(self, "make_" + name)()
            test_model.states.clear()
            test_model.states.from_file(os.path.join(directory, "states", name + ".nl"))
            self.assertModelEqual(base_model, test_model)

    def save(self, directory: str, version: typing.Optional[tuple[int, int]]):
        """Save all models to the given directory with the given serialization version."""
        os.makedirs(os.path.join(directory, "model"), exist_ok=True)
        os.makedirs(os.path.join(directory, "states"), exist_ok=True)

        for name in self.iter_names():
            print("saving", name)  # could add a verbose flag to toggle
            model = getattr(self, "make_" + name)()
            with open(os.path.join(directory, "model", name + ".nl"), "wb") as f:
                model.into_file(f, version=version, max_num_states=100)
            with open(os.path.join(directory, "states", name + ".nl"), "wb") as f:
                model.states.into_file(f, version=version)

    def test_model(self):
        """For all models, test the serializing and deserializing results in the same model."""
        for version in dwave.optimization._model.KNOWN_SERIALIZATION_VERSIONS:
            for name in self.iter_names():
                with self.subTest(version=version, model=name):
                    base_model = getattr(self, "make_" + name)()
                    with base_model.to_file(version=version, max_num_states=100) as f:
                        test_model = Model.from_file(f)
                    self.assertModelEqual(base_model, test_model)

    def test_states(self):
        """For all models, test the serializing and deserializing results in the same states."""
        for version in dwave.optimization._model.KNOWN_SERIALIZATION_VERSIONS:
            for name in self.iter_names():
                make = getattr(self, "make_" + name)

                with self.subTest(version=version, model=name):
                    base_model = getattr(self, "make_" + name)()
                    test_model = getattr(self, "make_" + name)()

                    test_model.states.clear()
                    with base_model.states.to_file(version=version) as f:
                        test_model.states.from_file(f)

                    self.assertModelEqual(base_model, test_model)

    def make_empty(self) -> Model:
        return Model()

    def make_ragged(self) -> Model:
        """A model with a ragged state"""
        model = Model()
        x = model.set(5)
        model.minimize(x.sum())

        model.states.resize(5)

        x.set_state(0, [0, 1])
        # no state at 1
        x.set_state(2, [])
        x.set_state(3, [0, 1, 2, 3, 4])
        x.set_state(4, [0, 1, 2, 3, 4])

        return model

    def make_same_shape(self) -> Model:
        """A model with a state of all the same shape"""
        model = Model()
        x = model.binary(5)
        model.minimize(x.sum())

        model.states.resize(5)

        x.set_state(0, [0, 0, 0, 0, 0])
        x.set_state(1, [0, 0, 0, 0, 1])
        x.set_state(2, [0, 0, 0, 1, 1])
        x.set_state(3, [0, 0, 1, 1, 1])
        x.set_state(4, [0, 1, 1, 1, 1])

        return model


if __name__ == "__main__":
    import argparse

    def as_directory(directory: str) -> str:
        if not os.path.isdir(directory):
            raise ValueError(f"{directory} does not exist")
        return directory

    def as_version(version: str) -> tuple[int, int]:
        return tuple(map(int, version.split(".")))

    parser = argparse.ArgumentParser(description='Save/load models for testing')

    parser.add_argument(
        "saveload",
        choices=("save", "load"),
        help="Whether to save or load the files.",
    )
    parser.add_argument(
        "directory",
        help=(
            "The directory to save/load the files. Must already exist. "
            "Files are overwritten when saving."
        ),
        type=as_directory,
    )
    parser.add_argument(
        "--version",
        help=(
            "The directory to save/load the files. Must already exist. "
            "Files are overwritten when saving."
        ),
        type=as_version,
    )
    args = parser.parse_args()

    tests = TestSerialization()
    if args.saveload == "save":
        tests.save(args.directory, version=args.version)
    elif args.saveload == "load":
        tests.load(args.directory)
    else:
        raise RuntimeError("unexpected saveload option")
