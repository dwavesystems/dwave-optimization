project(
    'dwave-optimization',
    'cpp', 'cython',
    version: '0.2.0',
    default_options: [
        'cpp_std=c++20',
    ],
)

py = import('python').find_installation(
    pure: false,
)

libdwave_optimization = shared_library(
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
    include_directories: include_directories('dwave/optimization/include/'),
    install: true,
    install_dir: py.get_install_dir(pure: false) / 'dwave/optimization/'
)

py.extension_module(
    'model',
    'dwave/optimization/model.pyx',
    subdir: 'dwave/optimization/',
    override_options : ['cython_language=cpp'],
    dependencies : py.dependency(),
    link_with: [libdwave_optimization],
    include_directories: include_directories('dwave/optimization/include/'),
    install: true,
)

py.extension_module(
    'symbols',
    'dwave/optimization/symbols.pyx',
    subdir: 'dwave/optimization/',
    override_options : ['cython_language=cpp'],
    dependencies : py.dependency(),
    link_with: [libdwave_optimization],
    include_directories: include_directories('dwave/optimization/include/'),
    install: true,
)

install_subdir('dwave', install_dir: py.get_install_dir(pure: false))
