
catch2 = subproject(
    'catch2',
    default_options: {'tests': false},  # disable Catch2's self-tests
).get_variable('catch2_with_main_dep')

tests_all = executable(
    'tests-all',
    [
        'nodes/indexing/test_advanced.cpp',
        'nodes/indexing/test_basic.cpp',
        'nodes/mathematical/test_binaryop.cpp',
        'nodes/mathematical/test_naryop.cpp',
        'nodes/mathematical/test_reduce.cpp',
        'nodes/mathematical/test_unaryop.cpp',
        'nodes/test_collections.cpp',
        'nodes/test_constants.cpp',
        'nodes/test_creation.cpp',
        'nodes/test_flow.cpp',
        'nodes/test_manipulation.cpp',
        'nodes/test_numbers.cpp',
        'nodes/test_quadratic_model.cpp',
        'test_array.cpp',
        'test_graph.cpp',
        'test_utils.cpp',
    ],
    dependencies: [
        catch2,
        libdwave_optimization,
    ],
)

test('tests-all', tests_all, verbose: true)
