
catch2 = subproject(
    'catch2',
    default_options: {'tests': false},  # disable Catch2's self-tests
).get_variable('catch2_with_main_dep')

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
        'tests/test_nodes_manipulation.cpp',
        'tests/test_nodes_mathematical.cpp',
        'tests/test_nodes_numbers.cpp',
        'tests/test_nodes_quadratic_model.cpp',
        'tests/test_utils.cpp',
    ],
    dependencies: [
        catch2,
        libdwave_optimization,
    ],
)

test('tests-all', tests_all, verbose: true)
