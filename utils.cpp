#include "utils.h"
#include <iostream>
#include <stdlib.h>

#include <immintrin.h>

constexpr int nvec_256 = 8;
constexpr int nvec_512 = 16;

#if defined(__AVX__)
#include "avx2_intrinsics.h"
inline void MAKE_VOLATILE_STRIDE(int a, int b) {}

namespace m256 {

using E = __m256;
using R = __m256;

inline int WS(const stride s, const stride i) { return s * i; }

#include "dft_c2cf_1024.c"
#include "dft_c2cf_128.c"
#include "dft_c2cf_256.c"
#include "dft_c2cf_32.c"
#include "dft_c2cf_512.c"
#include "dft_c2cf_64.c"
#include "dft_c2r_1024.c"
#include "dft_c2r_128.c"
#include "dft_c2r_256.c"
#include "dft_c2r_32.c"
#include "dft_c2r_512.c"
#include "dft_c2r_64.c"
#include "dft_r2cf_1024.c"
#include "dft_r2cf_128.c"
#include "dft_r2cf_256.c"
#include "dft_r2cf_32.c"
#include "dft_r2cf_512.c"
#include "dft_r2cf_64.c"

} // namespace m256

void avx2_c2c_gather(int fft_size, int batch_size, __m256 *simd_arr,
                     float *in_data) {
  int num_floats = 2 * fft_size;

  __m256i vIdx = _mm256_set_epi32(
      num_floats * 7, num_floats * 6, num_floats * 5, num_floats * 4,
      num_floats * 3, num_floats * 2, num_floats, 0);
  for (unsigned i = 0; i < batch_size / nvec_256; i++)
    for (unsigned j = 0; j < num_floats; j++) {
      float *ptr = in_data + j + i * num_floats * nvec_256;
      simd_arr[j + i * num_floats] =
          _mm256_i32gather_ps(static_cast<float *>(ptr), vIdx, 4);
    }
}

void avx2_c2c_scatter(int fft_size, int batch_size, __m256 *simd_arr,
                      float *out_data) {
  int num_floats = 2 * fft_size;
  for (unsigned i = 0; i < batch_size / nvec_256; i++)
    for (unsigned j = 0; j < num_floats; j++)
      for (unsigned k = 0; k < nvec_256; k++) {
        out_data[i * num_floats * nvec_256 + k * num_floats + j] =
            simd_arr[j + i * num_floats][k];
      }
}

void AVX2DFTc2c(const __m256 *realInput, const __m256 *imaginaryInput,
                __m256 *realOutput, __m256 *imaginaryOutput, int inputstride,
                int outputStride, int batch_size, int batchStrideIn,
                int batchStrideOut, int fft_size) {
  void (*foo)(const __m256 *realInput, const __m256 *imaginaryInput,
              __m256 *realOutput, __m256 *imaginaryOutput, int inputstride,
              int outputStride, int batch_size, int batchStrideIn,
              int batchStrideOut);
  switch (fft_size) {
  case 32:
    foo = &(m256::dft_codelet_c2cf_32);
    break;
  case 64:
    foo = &(m256::dft_codelet_c2cf_64);
    break;
  case 128:
    foo = &(m256::dft_codelet_c2cf_128);
    break;
  case 256:
    foo = &(m256::dft_codelet_c2cf_256);
    break;
  case 512:
    foo = &(m256::dft_codelet_c2cf_512);
    break;
  case 1024:
    foo = &(m256::dft_codelet_c2cf_1024);
    break;
  }
  for (int i = 0; i < batch_size / nvec_256; i++) {
    foo(realInput, imaginaryInput, realOutput, imaginaryOutput, inputstride,
        outputStride, 1, batchStrideIn, batchStrideOut);
#pragma omp atomic
    realInput = realInput + batchStrideIn;
#pragma omp atomic
    imaginaryInput = imaginaryInput + batchStrideIn;
#pragma omp atomic
    realOutput = realOutput + batchStrideOut;
#pragma omp atomic
    imaginaryOutput = imaginaryOutput + batchStrideOut;
  }
}
#undef DK
#endif

#if defined(__AVX512F__)

#include "avx512_intrinsics.h"

namespace m512 {

using E = __m512;
using R = __m512;

inline int WS(const stride s, const stride i) { return s * i; }

#include "dft_c2cf_1024.c"
#include "dft_c2cf_128.c"
#include "dft_c2cf_256.c"
#include "dft_c2cf_32.c"
#include "dft_c2cf_512.c"
#include "dft_c2cf_64.c"
#include "dft_c2r_1024.c"
#include "dft_c2r_128.c"
#include "dft_c2r_256.c"
#include "dft_c2r_32.c"
#include "dft_c2r_512.c"
#include "dft_c2r_64.c"
#include "dft_r2cf_1024.c"
#include "dft_r2cf_128.c"
#include "dft_r2cf_256.c"
#include "dft_r2cf_32.c"
#include "dft_r2cf_512.c"
#include "dft_r2cf_64.c"

} // namespace m512

void avx512_c2c_gather(int fft_size, int batch_size, __m512 *simd_arr,
                       float *in_data) {
  int num_floats = 2 * fft_size;

  __m512i vIdx = _mm512_set_epi32(
      num_floats * 15, num_floats * 14, num_floats * 13, num_floats * 12,
      num_floats * 11, num_floats * 10, num_floats * 9, num_floats * 8,
      num_floats * 7, num_floats * 6, num_floats * 5, num_floats * 4,
      num_floats * 3, num_floats * 2, num_floats, 0);
  for (unsigned i = 0; i < batch_size / nvec_512; i++)
    for (unsigned j = 0; j < num_floats; j++) {
      float *ptr = in_data + j + i * num_floats * nvec_512;
      simd_arr[j + i * num_floats] =
          _mm512_i32gather_ps(vIdx, static_cast<void *>(ptr), 4);
    }
}

void avx512_c2c_scatter(int fft_size, int batch_size, __m512 *simd_arr,
                        float *out_data) {
  int num_floats = 2 * fft_size;
  for (unsigned i = 0; i < batch_size / nvec_512; i++)
    for (unsigned j = 0; j < num_floats; j++)
      for (unsigned k = 0; k < nvec_512; k++) {
        out_data[i * num_floats * nvec_512 + k * num_floats + j] =
            simd_arr[j + i * num_floats][k];
      }
}

void AVX512DFTc2c(const __m512 *realInput, const __m512 *imaginaryInput,
                  __m512 *realOutput, __m512 *imaginaryOutput, int inputstride,
                  int outputStride, int batch_size, int batchStrideIn,
                  int batchStrideOut, int fft_size) {
  void (*foo)(const __m512 *realInput, const __m512 *imaginaryInput,
              __m512 *realOutput, __m512 *imaginaryOutput, int inputstride,
              int outputStride, int batch_size, int batchStrideIn,
              int batchStrideOut);
  switch (fft_size) {
  case 32:
    foo = &(m512::dft_codelet_c2cf_32);
    break;
  case 64:
    foo = &(m512::dft_codelet_c2cf_64);
    break;
  case 128:
    foo = &(m512::dft_codelet_c2cf_128);
    break;
  case 256:
    foo = &(m512::dft_codelet_c2cf_256);
    break;
  case 512:
    foo = &(m512::dft_codelet_c2cf_512);
    break;
  case 1024:
    foo = &(m512::dft_codelet_c2cf_1024);
    break;
  }
#pragma omp parallel for
  for (int i = 0; i < batch_size / nvec_512; i++) {
    foo(realInput, imaginaryInput, realOutput, imaginaryOutput, inputstride,
        outputStride, 1, batchStrideIn, batchStrideOut);
#pragma omp atomic
    realInput = realInput + batchStrideIn;
#pragma omp atomic
    imaginaryInput = imaginaryInput + batchStrideIn;
#pragma omp atomic
    realOutput = realOutput + batchStrideOut;
#pragma omp atomic
    imaginaryOutput = imaginaryOutput + batchStrideOut;
  }
}
#undef DK
#endif

void FFTcDFTc2c(const double *realInput, const double *imaginaryInput,
                double *realOutput, double *imaginaryOutput, int inputstride,
                int outputStride, int batch_size, int batchStrideIn,
                int batchStrideOut, int fft_size) {
  void (*foo)(double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
  switch (fft_size) {

  case 64:
    foo = &(fft_memref_8_64_inter_cpu_novec);
    break;
  case 128:
    foo = &(fft_memref_2_128_inter_cpu_novec);
    break;
  case 256:
    foo = &(fft_memref_16_256_inter_cpu_novec);
    break;
  case 512:
    foo = &(fft_memref_8_512_inter_cpu_novec);
    break;
  case 1024:
    foo = &(fft_memref_4_1024_inter_cpu_novec);
    break;
  }
  for (int i = 0; i < batch_size; i++) {
    foo((double *)realInput, (double *)realInput, 0, 1, 2, 3, 4, 5, 1,
        (double *)realOutput, (double *)realOutput, 0, 1, 2, 3, 4, 5, 1);
  }
}