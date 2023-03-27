#!/bin/bash
ml load buildenv-intel/2018a-eb
ml load CMake/3.23.2

cmake .. -DCMAKE_BUILD_TYPE=Debug   -DFFTW_INCLUDE_DIR=/proj/snic2022-22-1035/sc/fftw/fftw-3.3.10/include -DFFTW_LIBRARY=/proj/snic2022-22-1035/sc/fftw/fftw-3.3.10/lib/libfftw3.a
make -j12
