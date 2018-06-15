#!/bin/bash

if [ "$(uname)" == "Darwin" ]
then
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

if [ $CONDA_PY -gt 30 ]; then
    pip install --no-deps codacy-coverage
fi

$PYTHON setup.py install --prefix=$PREFIX
