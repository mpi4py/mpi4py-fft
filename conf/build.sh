#!/bin/bash

if [ "$(uname)" == "Darwin" ]
then
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

$PYTHON setup.py install --prefix=$PREFIX
