#!/bin/bash

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release   -DFFTW_INCLUDE_DIR=/proj/snic2022-22-1035/sc/fftw/fftw-3.3.10/include -DFFTW_LIBRARY=/proj/snic2022-22-1035/sc/fftw/fftw-3.3.10/lib/libfftw3.a
make
cd ..
./build/dft_simd --gtest_filter=*erf*