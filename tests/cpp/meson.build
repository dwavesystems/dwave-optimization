
catch2 = subproject(
    'catch2',
    default_options: {'tests': false},  # disable Catch2's self-tests
).get_variable('catch2_with_main_dep')

tests_all = executable(
    'tests-all',
    [
        'nodes/indexing/test_advanced.cpp',
        'nodes/indexing/test_basic.cpp',
        'nodes/test_binaryop.cpp',
        'nodes/test_collections.cpp',
        'nodes/test_constants.cpp',
        'nodes/test_creation.cpp',
        'nodes/test_flow.cpp',
        'nodes/test_inputs.cpp',
        'nodes/test_interpolation.cpp',
        'nodes/test_lp.cpp',
        'nodes/test_lambda.cpp',
        'nodes/test_manipulation.cpp',
        'nodes/test_naryop.cpp',
        'nodes/test_numbers.cpp',
        'nodes/test_quadratic_model.cpp',
        'nodes/test_reduce.cpp',
        'nodes/test_softmax.cpp',
        'nodes/test_sorting.cpp',
        'nodes/test_statistics.cpp',
        'nodes/test_unaryop.cpp',
        'test_array.cpp',
        'test_double_kahan.cpp',
        'test_fraction.cpp',
        'test_functional.cpp',
        'test_graph.cpp',
        'test_iterators.cpp',
        'test_simplex.cpp',
        'test_type_list.cpp',
    ],
    dependencies: [
        catch2,
        libdwave_optimization,
    ],
    include_directories: include_directories('../../dwave/optimization/src/'),
)

test('tests-all', tests_all, verbose: true)
