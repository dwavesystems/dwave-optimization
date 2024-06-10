# Copyright 2023 D-Wave Systems Inc.
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

import dwave.optimization.generators

# monkeypatch the cloud-client to add NL support
import dwave.optimization.client

from dwave.optimization.model import Model, _register_node_subclasses
from dwave.optimization.mathematical import *

__version__ = "0.0.1.dev8"


def get_include():
    """Return the directory with dwave-optimization's header files."""
    import os.path
    return os.path.join(os.path.dirname(__file__), 'include')


def get_library_dir():
    """Return a list of all of the source files."""
    import os.path
    import platform
    if platform.system() == "Windows":
        raise RuntimeError("dwave-optimization does not distribute a library on Windows")
    return os.path.dirname(__file__)


def get_library():
    """Return the shared library name."""
    import platform
    if platform.system() == "Windows":
        raise RuntimeError("dwave-optimization does not distribute a library on Windows")
    return "dwave-optimization"


_register_node_subclasses()
del _register_node_subclasses
