WHEEL="$1"
DEST_DIR="$2"
OPENBLAS_DIR=$(python -c"import scipy_openblas64 as sop; print(sop.get_lib_dir())")

delvewheel repair --add-path $OPENBLAS_DIR --namespace-pkg dwave -w $DEST_DIR $WHEEL
