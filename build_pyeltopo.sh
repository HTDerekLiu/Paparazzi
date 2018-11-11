#!/bin/bash


OPTIND=1

while getopts "p:ih" opt; do

    echo "opt: $opt"
    case "$opt" in
        h)
            echo "USAGE: build_pyeltopo.sh [-i] [-p pythonbin]"
            ;;
        p)
            echo "P FOUND"
            PYTHON_EXEC="$OPTARG"
            echo $OPTARG
            echo "pyexec: $PYTHON_EXEC"
            ;;
        i)
            DO_INSTALL="yes"
            ;;
            
            \?)
                echo "Unknowno option ($opt)"
            ;;
    esac
    
done
: ${PYTHON_EXEC:="$( which python )"}


mkdir build -p
pushd build
cmake ../extern/eltopo -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$PYTHON_EXEC
make -j9
if [ $DO_INSTALL ]; then
    make install
fi
popd
