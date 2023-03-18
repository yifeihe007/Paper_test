#ifndef UTILS_H
#define UTILS_H

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

enum class FFT_type { C2C, R2C, C2R };

#if defined(__AVX__)

void avx2_c2c_gather(int fft_size, int batch_size, __m256 *simd_arr,
                     float *in_data);
void avx2_c2c_scatter(int fft_size, int batch_size, __m256 *simd_arr,
                      float *out_data);
void AVX2DFTc2c(const __m256 *realInput, const __m256 *imaginaryInput,
                __m256 *realOutput, __m256 *imaginaryOutput, int inputstride,
                int outputStride, int batch_size, int batchStrideIn,
                int batchStrideOut, int fft_size);

#endif
#if defined(__AVX512F__)
void avx512_c2c_gather(int fft_size, int batch_size, __m512 *simd_arr,
                       float *in_data);
void avx512_c2c_scatter(int fft_size, int batch_size, __m512 *simd_arr,
                        float *out_data);
void AVX512DFTc2c(const __m512 *realInput, const __m512 *imaginaryInput,
                  __m512 *realOutput, __m512 *imaginaryOutput, int inputstride,
                  int outputStride, int batch_size, int batchStrideIn,
                  int batchStrideOut, int fft_size);
#endif
#endif

void FFTcDFTc2c(const double *realInput, const double *imaginaryInput,
                double *realOutput, double *imaginaryOutput, int inputstride,
                int outputStride, int batch_size, int batchStrideIn,
                int batchStrideOut, int fft_size);

extern "C" void
fft_memref(double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
           intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
           intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
           // Second Memref (%arg1)
           double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
           intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
           intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);

extern "C" void
fft_memref_2_4(double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
               intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
               intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
               // Second Memref (%arg1)
               double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
               intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
               intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);

extern "C" void fft_memref_16_256_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_16_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_128_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_256_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_512_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_1024_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_2048_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_2_4096_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_4_16_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_4_64_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_4_256_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_4_1024_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_4_4096_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_8_64_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_8_512_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_8_4096_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_16_256_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);
extern "C" void fft_memref_16_4096_inter_cpu_novec(
    double *allocated_ptr0, double *aligned_ptr0, intptr_t offset0,
    intptr_t size0_d0, intptr_t size0_d1, intptr_t size0_d2,
    intptr_t stride0_d0, intptr_t stride0_d1, intptr_t stride0_d2,
    // Second Memref (%arg1)
    double *allocated_ptr1, double *aligned_ptr1, intptr_t offset1,
    intptr_t size1_d0, intptr_t size1_d1, intptr_t size1_d2,
    intptr_t stride1_d0, intptr_t stride1_d1, intptr_t stride1_d2);