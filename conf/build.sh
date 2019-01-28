#!/bin/bash

if [ "$(uname)" == "Darwin" ]
then
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
