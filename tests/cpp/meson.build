
catch2 = subproject(
    'catch2',
    default_options: {'tests': false},  # disable Catch2's self-tests
).get_variable('catch2_with_main_dep')

tests_all = executable(
    'tests-all',
    [
        'tests/nodes/indexing/test_advanced.cpp',
        'tests/nodes/indexing/test_basic.cpp',
        'tests/nodes/mathematical/test_binaryop.cpp',
        'tests/nodes/mathematical/test_naryop.cpp',
        'tests/nodes/mathematical/test_reduce.cpp',
        'tests/nodes/mathematical/test_unaryop.cpp',
        'tests/nodes/test_collections.cpp',
        'tests/nodes/test_constants.cpp',
        'tests/nodes/test_flow.cpp',
        'tests/nodes/test_manipulation.cpp',
        'tests/nodes/test_numbers.cpp',
        'tests/nodes/test_quadratic_model.cpp',
        'tests/test_array.cpp',
        'tests/test_examples.cpp',
        'tests/test_graph.cpp',
        'tests/test_utils.cpp',
    ],
    dependencies: [
        catch2,
        libdwave_optimization,
    ],
)

test('tests-all', tests_all, verbose: true)
