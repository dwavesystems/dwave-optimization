#!/usr/bin/env python3

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
import os.path
import sys

try:
    import scipy_openblas64
except ImportError:
    sys.exit()

# meson needs the include_dir to be relative
print(os.path.relpath(scipy_openblas64.get_include_dir(), os.getcwd()))

# But the actual library path needs to be absolute
print(scipy_openblas64.get_lib_dir())

# This one isn't a path at all
print(scipy_openblas64.get_library())
