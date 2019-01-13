#!/bin/bash


OPTIND=1

while getopts "p:ih" opt; do

    echo "opt: $opt"
    case "$opt" in
        h)
            echo "USAGE: build_pyeltopo.sh [-b] [-p pythonbin]"
            echo "[-b] build only, do not install"
            echo "[-p pythonbin] use pythonbin as the python binary"
            ;;
        p)
            echo "P FOUND"
            PYTHON_EXEC="$OPTARG"
            echo $OPTARG
            echo "pyexec: $PYTHON_EXEC"
            ;;
        b)
            DO_INSTALL="no"
            ;;
            
            \?)
                echo "Unknown option ($opt)"
            ;;
    esac
    
done
: ${PYTHON_EXEC:="$( which python )"}

pushd extern/pyeltopo
if [ $DO_INSTALL != no ]; then
echo $PYTHON_EXEC setup.py install --user
$PYTHON_EXEC setup.py install --user --prefix=
else
$PYTHON_EXEC setup.py build
fi
#if [ $DO_INSTALL ]; then
#    make install
#fi

#mkdir build -p
#pushd build
#cmake ../extern/eltopo -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$PYTHON_EXEC
#make -j9
#if [ $DO_INSTALL ]; then
#    make install
#fi
#popd
