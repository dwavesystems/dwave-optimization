project(
    'dwave-optimization',
    'c', 'cpp', 'cython',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',  # add -DNDEBUG when building for "release", which we do by default
        'optimization=3',
        'pkgconfig.relocatable=true',
        'warning_level=1',  # equivalent to -Wall. We should consider cranking it up to 3
    ],

    # As of August 2024, there is no nicer way that I know of to dynamically get the version
    version: run_command(
        ['dwave/optimization/_build/_version.py'],
        capture: true,
        check: true,
    ).stdout(),
)

# We want some debugging symbols
add_project_arguments(['-g1'], language: ['cpp'])

py = import('python').find_installation(pure: false)

dwave_optimization_include = include_directories('dwave/optimization/include/')
dwave_optimization_src = [
    'dwave/optimization/src/nodes/collections.cpp',
    'dwave/optimization/src/nodes/constants.cpp',
    'dwave/optimization/src/nodes/creation.cpp',
    'dwave/optimization/src/nodes/flow.cpp',
    'dwave/optimization/src/nodes/indexing.cpp',
    'dwave/optimization/src/nodes/inputs.cpp',
    'dwave/optimization/src/nodes/interpolation.cpp',
    'dwave/optimization/src/nodes/lp.cpp',
    'dwave/optimization/src/nodes/manipulation.cpp',
    'dwave/optimization/src/nodes/mathematical.cpp',
    'dwave/optimization/src/nodes/numbers.cpp',
    'dwave/optimization/src/nodes/quadratic_model.cpp',
    'dwave/optimization/src/nodes/testing.cpp',
    'dwave/optimization/src/array.cpp',
    'dwave/optimization/src/graph.cpp',
    'dwave/optimization/src/simplex.cpp',
    'dwave/optimization/src/utils.cpp',
]

if host_machine.system() == 'windows'
    # Shared libraries don't work so well on windows, so we build from a static
    # library instead
    libdwave_optimization = declare_dependency(
        link_with: static_library(
            'dwave-optimization',
            sources: dwave_optimization_src,
            include_directories: dwave_optimization_include,
        ),
        include_directories: dwave_optimization_include
    )
else
    # On unix we build/distribute a shared library
    libdwave_optimization = declare_dependency(
        link_with: shared_library(
            'dwave-optimization',
            sources: dwave_optimization_src,
            include_directories: dwave_optimization_include,
            install: true,
            install_dir: py.get_install_dir(pure: false) / 'dwave/optimization/',
        ),
        include_directories: dwave_optimization_include
    )
endif

# Cython allows the creation of a "shared" file for some of their datastructures
# See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#shared-module
py.extension_module(
    '_cyutility',
    custom_target(
        output: '_cyutility.c',
        command: [meson.get_compiler('cython').cmd_array()[0], '--generate-shared', '@OUTPUT0@'],
    ),
    install: true,
    install_rpath: '$ORIGIN',
    subdir: 'dwave/optimization/',
)

py.extension_module(
    '_model',
    'dwave/optimization/_model.pyx',
    dependencies: libdwave_optimization,
    gnu_symbol_visibility: 'default',
    install: true,
    install_rpath: '$ORIGIN',
    override_options : ['cython_language=cpp'],
    cython_args : ['--shared=dwave.optimization._cyutility'],
    subdir: 'dwave/optimization/',
)

py.extension_module(
    'states',
    'dwave/optimization/states.pyx',
    dependencies: libdwave_optimization,
    gnu_symbol_visibility: 'default',
    install: true,
    install_rpath: '$ORIGIN',
    override_options : ['cython_language=cpp'],
    cython_args : ['--shared=dwave.optimization._cyutility'],
    subdir: 'dwave/optimization/',
)

py.extension_module(
    'symbols',
    'dwave/optimization/symbols.pyx',
    dependencies: libdwave_optimization,
    gnu_symbol_visibility: 'default',
    install: true,
    install_rpath: '$ORIGIN',
    override_options : ['cython_language=cpp'],
    cython_args : ['--shared=dwave.optimization._cyutility'],
    subdir: 'dwave/optimization/',
)

py.extension_module(
    '_utilities',
    'dwave/optimization/_utilities.pyx',
    gnu_symbol_visibility: 'default',
    install: true,
    install_rpath: '$ORIGIN',
    override_options : ['cython_language=cpp'],
#    cython_args : ['--shared=dwave.optimization._cyutility'],  # throws compiler warnings as of May 2025
    subdir: 'dwave/optimization/',
)

install_subdir('dwave/optimization', install_dir: py.get_install_dir(pure: false) / 'dwave')

# meson doesn't disable tests by default, so we let the user decide via the build-tests
# option. If set to 'auto', then we build the tests for non-release builds.
if get_option('build-tests').enabled() or (get_option('build-tests').auto() and get_option('buildtype') != 'release')
    subdir('tests/cpp/')
endif
