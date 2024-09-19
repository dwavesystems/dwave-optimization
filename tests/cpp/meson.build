tests_all = executable(
    'tests-all',
    [
        'tests/test_array.cpp',
        'tests/test_examples.cpp',
        'tests/test_graph.cpp',
        'tests/test_nodes_collections.cpp',
        'tests/test_nodes_constants.cpp',
        'tests/test_nodes_flow.cpp',
        'tests/test_nodes_indexing.cpp',
        'tests/test_nodes_mathematical.cpp',
        'tests/test_nodes_numbers.cpp',
        'tests/test_nodes_quadratic_model.cpp',
        'tests/test_utils.cpp',
        'test_main.cpp',
    ],
    dependencies: [
        subproject('catch2').get_variable('catch2_dep'),
        libdwave_optimization,
    ],
)

test('tests-all', tests_all, verbose: true)
