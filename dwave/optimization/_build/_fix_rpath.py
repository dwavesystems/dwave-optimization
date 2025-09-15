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

# Makes a few assumptions which (as of Sept 2025) are true in cibuildwheel
# - code is in /project/
# - patchelf is available

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile


def parse_arguments() -> pathlib.Path:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wheel",
        type=is_wheel,
        help="The absolute path to the built wheel",
    )
    return parser.parse_args().wheel


def is_wheel(path):
    # Quick sanity check, certainly not inclusive
    path = pathlib.Path(path)
    if pathlib.Path.is_file(path) and path.suffix == ".whl":
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a wheel")


if __name__ == "__main__":
    wheel: pathlib.Path = parse_arguments()

    if sys.platform.startswith("linux"):
        platform = "linux"
    elif sys.platform.startswith("darwin"):
        platform = "osx"
    else:
        raise NotImplementedError("_fix_rpath.py only supports OSX and linux")

    # Do most of it inside a temporary directory
    with tempfile.TemporaryDirectory(prefix="fix-rpath-") as tmpdir:
        zipdir = pathlib.Path(tmpdir)

        # Unzip the wheel into the temporary directory
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(zipdir)

        # Get the path to libdwave-optimization. The extension differs for linux/osx
        lib, = zipdir.glob(
            f"dwave/optimization/libdwave-optimization.{'so' if platform == 'linux' else 'dylib'}",
        )

        # We want to repair the rpath on all singly nested shared objects
        # We could generalize this to arbitrarily nested, but this is simpler
        # for now
        # In this case the extensions are always .so, whether linux or osx
        for nested_so in zipdir.glob("dwave/optimization/*/*.so"):

            if platform == "linux":
                command = f"patchelf --set-rpath '$ORIGIN/../' {nested_so}"
            else:
                command = (
                    "install_name_tool "
                    "-change @rpath/libdwave-optimization.dylib @rpath/../libdwave-optimization.dylib "
                    f"{nested_so}"
                )

            subprocess.check_call(command, shell=True)

        # Now repack the wheel into a zipfile
        newwheel = pathlib.Path(shutil.make_archive(wheel.stem, "zip", zipdir))

        # And finally do some swapsies, saving the old one for posterity
        pathlib.Path.mkdir(wheel.parent / "old", exist_ok=True)
        pathlib.Path.rename(wheel, wheel.parent / "old" / wheel.name)
        pathlib.Path.rename(newwheel, wheel)
