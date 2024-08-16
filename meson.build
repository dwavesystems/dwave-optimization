project(
    'dwave-optimization',
    'cpp', 'cython',
    version: '0.2.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',  # add -DNDEBUG when building for "release", which we do by default
        'pkgconfig.relocatable=true',
    ],
)
add_project_arguments(['-g1'], language: ['cpp'])

py = import('python').find_installation(
    pure: false,
)

dwave_optimization_include = include_directories('dwave/optimization/include/')

libdwave_optimization = library(
    'dwave-optimization',
    sources: [
        'dwave/optimization/src/nodes/collections.cpp',
        'dwave/optimization/src/nodes/flow.cpp',
        'dwave/optimization/src/nodes/indexing.cpp',
        'dwave/optimization/src/nodes/mathematical.cpp',
        'dwave/optimization/src/nodes/numbers.cpp',
        'dwave/optimization/src/nodes/quadratic_model.cpp',
        'dwave/optimization/src/nodes/testing.cpp',
        'dwave/optimization/src/array.cpp',
        'dwave/optimization/src/graph.cpp',
        'dwave/optimization/src/utils.cpp',
        'dwave/optimization/src/vartypes.cpp',
    ],
    include_directories: dwave_optimization_include,
    install: true,
    install_dir: py.get_install_dir(pure: false) / 'dwave/optimization/'
)

why_does_this_help = declare_dependency(
    link_with: libdwave_optimization,
)

py.extension_module(
    'model',
    'dwave/optimization/model.pyx',
    subdir: 'dwave/optimization/',
    override_options : ['cython_language=cpp'],
    #dependencies : py.dependency(),
    #link_with: [libdwave_optimization],
    dependencies: [why_does_this_help],
    include_directories: dwave_optimization_include,
    install: true,
      install_rpath: '$ORIGIN',
)

py.extension_module(
    'symbols',
    'dwave/optimization/symbols.pyx',
    subdir: 'dwave/optimization/',
    override_options : ['cython_language=cpp'],
    #dependencies : py.dependency(),
    #link_with: [libdwave_optimization],
    dependencies: [why_does_this_help],
    include_directories: dwave_optimization_include,
    install: true,
  install_rpath: '$ORIGIN',
)

install_subdir('dwave', install_dir: py.get_install_dir(pure: false))
