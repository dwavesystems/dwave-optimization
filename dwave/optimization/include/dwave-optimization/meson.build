# Generate config.hpp which includes information about the compilers/system
# we're being built on

conf = configuration_data()

conf.set(
    'DWOPT_CXXABI_HAS_DEMANGLE',
    cpp.has_type('abi::__cxa_demangle', prefix: '#include <cxxabi.h>').to_int(),
)

conf.set('DWOPT_HAS_BLAS', has_blas.to_int())

conf.set(
    'DWOPT_SYS_TYPES_HAS_SSIZE_T',
    cpp.has_type('ssize_t', prefix: '#include <sys/types.h>').to_int(),
)

configure_file(
    output: 'config.hpp',
    configuration: conf,
    install: true,
    install_dir: py.get_install_dir(pure: false) / 'dwave/optimization/include/dwave-optimization/',
)
