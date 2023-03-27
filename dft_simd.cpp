/*

   dft_simd/dft_simd.cpp -- Stephen Fegan -- 2018-02-19

   Test drive for FFTW speed tests and for SIMD genfft codelets

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole Polytechnique, CNRS/IN2P3

   This file is part of "dft_simd"

   "dft_simd" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "dft_simd" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#include "utils.h"
#include <chrono>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <sys/time.h>

#define N 1
#define M 256
#define L 2
const int Rounds = 30;
constexpr int nvec = 8;
constexpr int nvec_512 = 16;
constexpr int Iter = 10000;

double calculateSD(double data[]) {
  double sum = 0.0, mean, standardDeviation = 0.0;
  int i;

  for (i = 0; i < Rounds; ++i) {
    sum += data[i];
  }

  mean = sum / Rounds;

  for (i = 0; i < Rounds; ++i) {
    standardDeviation += pow(data[i] - mean, 2);
  }
  return sqrt(standardDeviation / Rounds);
}

double calculateMean(double data[]) {
  double sum = 0.0, mean = 0.0;
  int i;

  for (i = 0; i < Rounds; ++i) {
    sum += data[i];
  }

  mean = sum / Rounds;

  return mean;
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

TEST(TestDFT, manyc2cFFTW_Aligned_One) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = 1;
  double elaps[Rounds];
  for (int k = 0; k < Rounds; k++) {

    // fftw_init_threads();
    // fftw_plan_with_nthreads(ompT);

    fftw_complex *xt = fftw_alloc_complex(nsamp * nloop);
    fftw_complex *xf = fftw_alloc_complex(nsamp * nloop);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    // Store values in vector
    std::vector<double> values(2 * nsamp * nloop);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = dist(generator);
    }
    for (size_t i = 0; i < values.size(); i += 2) {
      xt[i / 2][0] = values[i];
      xt[i / 2][1] = values[i + 1];
    }

    fftw_plan plan =
        fftw_plan_many_dft(1, &nsamp, nloop, xt, nullptr, 1, nsamp, xf, nullptr,
                           1, nsamp, FFTW_FORWARD, FFTW_EXHAUSTIVE);

    using namespace std::chrono;
    high_resolution_clock::time_point iStart = high_resolution_clock::now();

    for (int i = 0; i < Iter; i++)
      fftw_execute(plan);

    high_resolution_clock::time_point iFinished = high_resolution_clock::now();
    duration<double, std::milli> iElaps = iFinished - iStart;
    fftw_destroy_plan(plan);
    // fftw_cleanup_threads();

    free(xt);
    free(xf);
    elaps[k] = iElaps.count();
  }

  for (int k = 0; k < Rounds; k++)
    std::cout << "round " << k << ": " << elaps[k] << std::endl;

  double sd = calculateSD(elaps);
  double mean = calculateMean(elaps);

  std::cout << "Ec2cFFTW : nsamp : " << nsamp << " nloop : " << nloop
            << " ompT : " << ompT << " kernel mean : " << mean << " sd : " << sd
            << " .\n ";

  // omp_destroy_lock(&writelock);
}
/*
TEST(TestDFT, manyr2cFFTW_Aligned_One) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  fftw_init_threads();
  fftw_plan_with_nthreads(ompT);

  // omp_lock_t writelock;
  // omp_init_lock(&writelock);

  double *xt = fftw_alloc_real((nsamp / 2 + 1) * 2 * nloop);
  double *xf = fftw_alloc_real((nsamp / 2 + 1) * 2 * nloop);

  fftw_plan plan = fftw_plan_many_dft_r2c(
      1, &nsamp, nloop, xt, nullptr, 1, (nsamp / 2 + 1) * 2,
      (fftw_complex *)xf, nullptr, 1, nsamp / 2 + 1, FFTW_MEASURE);

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++)
    fftw_execute(plan);
  double iElaps = cpuSecond() - iStart;
  printf("Er2cFFTW : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  fftw_destroy_plan(plan);
  fftw_cleanup_threads();
  fftw_free(xf);
  fftw_free(xt);

  // omp_destroy_lock(&writelock);
}

TEST(TestDFT, manyc2rFFTW_Aligned_One) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  fftw_init_threads();
  fftw_plan_with_nthreads(ompT);

  // omp_lock_t writelock;
  // omp_init_lock(&writelock);

  double *xt = fftw_alloc_real((nsamp / 2 + 1) * 2 * nloop);
  double *xf = fftw_alloc_real((nsamp / 2 + 1) * 2 * nloop);

  fftw_plan plan = fftw_plan_many_dft_c2r(
      1, &nsamp, nloop, (fftw_complex *)xt, nullptr, 1, (nsamp / 2 + 1), xf,
      nullptr, 1, (nsamp / 2 + 1) * 2, FFTW_MEASURE);

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++)
    fftw_execute(plan);
  double iElaps = cpuSecond() - iStart;
  printf("Ec2rFFTW : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  fftw_destroy_plan(plan);
  fftw_cleanup_threads();
  fftw_free(xf);
  fftw_free(xt);

  // omp_destroy_lock(&writelock);
}*/

#if defined(__AVX__)

/*
TEST(TestDFT, AVX2r2c) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();

  int num = (nsamp / 2 + 1) * 2;

  __m256 *xt = nullptr;
  __m256 *xf = nullptr;
  double *in_data = fftw_alloc_real(num * nloop);
  double *out_data = fftw_alloc_real(num * nloop);
  ::posix_memalign((void **)&xt, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));
  ::posix_memalign((void **)&xf, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));

  double iStart = cpuSecond();

  __m256i vIdx = _mm256_set_epi32(num * 7, num * 6, num * 5, num * 4, num * 3,
                                  num * 2, num, 0);

  for (unsigned i = 0; i < nloop / nvec; i++)

    for (unsigned j = 0; j < num; j++) {

      xt[j + i * num] = _mm256_i32gather_ps(
          (static_cast<double *>(in_data) + j + i * num * nvec), vIdx, 4);
    }

  double afterGather = cpuSecond();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m256::dft_codelet_r2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 64:
      m256::dft_codelet_r2cf_64(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 128:
      m256::dft_codelet_r2cf_128(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 256:
      m256::dft_codelet_r2cf_256(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 512:
      m256::dft_codelet_r2cf_512(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 1024:
      m256::dft_codelet_r2cf_1024(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                  num, num);
      break;
    }
  }
  double afterCodelet = cpuSecond();

  for (unsigned i = 0; i < nloop / nvec; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec; k++) {
        static_cast<double *>(out_data)[i * num * nvec + k * num + j] =
            xf[j + i * num][k];
      }
  double afterScatter = cpuSecond();
  double gather = afterGather - iStart;
  double codelet = afterCodelet - afterGather;
  double scatter = afterScatter - afterCodelet;
  printf("EAVX2r2c : nsamp : %d nloop : %d ompT : %d gather : %f codelet : %f "
         "scatter : %f \n",
         nsamp, nloop, ompT, gather * 1000, codelet * 1000, scatter * 1000);

  ::free(xf);
  ::free(xt);
  fftw_free(in_data);
  fftw_free(out_data);
}
TEST(TestDFT, AVX2c2r) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  int num = (nsamp / 2 + 1) * 2;

  __m256 *xt = nullptr;
  __m256 *xf = nullptr;
  double *in_data = fftw_alloc_real(num * nloop);
  double *out_data = fftw_alloc_real(num * nloop);
  ::posix_memalign((void **)&xt, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));
  ::posix_memalign((void **)&xf, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));

  double iStart = cpuSecond();

  __m256i vIdx = _mm256_set_epi32(num * 7, num * 6, num * 5, num * 4, num * 3,
                                  num * 2, num, 0);

  for (unsigned i = 0; i < nloop / nvec; i++)

    for (unsigned j = 0; j < num; j++) {

      xt[j + i * num] = _mm256_i32gather_ps(
          (static_cast<double *>(in_data) + j + i * num * nvec), vIdx, 4);
    }

  double afterGather = cpuSecond();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m256::dft_codelet_c2r_32(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                               num, num);
      break;
    case 64:
      m256::dft_codelet_c2r_64(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                               num, num);
      break;
    case 128:
      m256::dft_codelet_c2r_128(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 256:
      m256::dft_codelet_c2r_256(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 512:
      m256::dft_codelet_c2r_512(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 1024:
      m256::dft_codelet_c2r_1024(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    }
  }
  double afterCodelet = cpuSecond();
  for (unsigned i = 0; i < nloop / nvec; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec; k++) {
        static_cast<double *>(out_data)[i * num * nvec + k * num + j] =
            xf[j + i * num][k];
      }

  double afterScatter = cpuSecond();
  double gather = afterGather - iStart;
  double codelet = afterCodelet - afterGather;
  double scatter = afterScatter - afterCodelet;
  printf("EAVX2c2r : nsamp : %d nloop : %d ompT : %d gather : %f codelet : %f "
         "scatter : %f \n",
         nsamp, nloop, ompT, gather * 1000, codelet * 1000, scatter * 1000);
  ::free(xf);
  ::free(xt);
  fftw_free(in_data);
  fftw_free(out_data);
}*/

#endif

#if defined(__AVX512F__)
/*
TEST(TestDFT, AVX512r2c) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  // for (int i = 0; i < Iter; i++) {
  __m512 *xt = nullptr;
  __m512 *xf = nullptr;
  ::posix_memalign((void **)&xt, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));
  ::posix_memalign((void **)&xf, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));

  int ompT = omp_get_max_threads();

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m512::dft_codelet_r2cf_32(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 64:
      m512::dft_codelet_r2cf_64(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 128:
      m512::dft_codelet_r2cf_128(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 256:
      m512::dft_codelet_r2cf_256(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 512:
      m512::dft_codelet_r2cf_512(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 1024:
      m512::dft_codelet_r2cf_1024(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                  (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    }
  }
  double iElaps = cpuSecond() - iStart;
  printf("EAVX512r2c : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  ::free(xf);
  ::free(xt);
}

TEST(TestDFT, AVX512c2r) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  // for (int i = 0; i < Iter; i++) {
  __m512 *xt = nullptr;
  __m512 *xf = nullptr;
  ::posix_memalign((void **)&xt, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));
  ::posix_memalign((void **)&xf, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(double));

  int ompT = omp_get_max_threads();

  double iStart = cpuSecond();
  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m512::dft_codelet_c2r_32(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                               (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 64:
      m512::dft_codelet_c2r_64(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                               (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 128:
      m512::dft_codelet_c2r_128(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 256:
      m512::dft_codelet_c2r_256(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 512:
      m512::dft_codelet_c2r_512(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 1024:
      m512::dft_codelet_c2r_1024(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    }
  }
  double iElaps = cpuSecond() - iStart;
  printf("EAVX512c2r : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  ::free(xf);
  ::free(xt);
}*/

#endif
/*
TEST(TestDFT, correctness_fftc1) {
  // c2c
  int fft_size_init[1] = {256};
  int batch_size_init[1] = {1};
  double arg0[L][M][N];
  double arg1[L][M][N];
  for (size_t i = 0; i < sizeof(fft_size_init) / sizeof(fft_size_init[0]); i++)
    for (size_t j = 0; j < sizeof(batch_size_init) / sizeof(batch_size_init[0]);
         j++) {
      int fft_size = fft_size_init[i];
      int batch_size = batch_size_init[j];
      // __m512 *xt = nullptr;
      //__m512 *xf = nullptr;
      // size_t byte_size = 2 * fft_size * batch_size * sizeof(double);

      // ::posix_memalign((void **)(&xt), 64, byte_size);
      // ::posix_memalign((void **)(&xf), 64, byte_size);

      std::random_device rd;
      std::mt19937 generator(rd());
      std::uniform_real_distribution<double> dist(-10.0, 10.0);
      // Store values in vector
      std::vector<double> values(2 * fft_size * batch_size);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = dist(generator);
        //   values[i] = 1;
      }
      fftw_complex *xt_fftw = fftw_alloc_complex(fft_size * batch_size);
      fftw_complex *xf_fftw = fftw_alloc_complex(fft_size * batch_size);
      // fftw_complex *xt_fftc = fftw_alloc_complex(fft_size * batch_size);
      // fftw_complex *xf_fftc = fftw_alloc_complex(fft_size * batch_size);

      fftw_plan plan = fftw_plan_many_dft(
          1, &fft_size, batch_size, xt_fftw, &fft_size, 1, fft_size, xf_fftw,
          &fft_size, 1, fft_size, FFTW_FORWARD, FFTW_MEASURE);

      for (size_t i = 0; i < values.size(); i += 2) {
        xt_fftw[i / 2][0] = values[i];
        xt_fftw[i / 2][1] = values[i + 1];
      }

      fftw_execute(plan);

      for (int i = 0; i < L; i++) {
        for (int j = 0; j < M; j++) {
          for (int k = 0; k < N; k++) {

            arg0[i][j][k] = values[k * 256 + 2 * j + i];
            arg1[i][j][k] = 0;
          }
        }
      }
      fft_memref((double *)arg0, (double *)arg0, 0, L, M, N, L, M, 1,
                 (double *)arg1, (double *)arg1, 0, L, M, N, L, M, 1);

      // std::vector<double> out_array(2 * fft_size * batch_size);
      // avx512_c2c_gather(fft_size, batch_size, xt, &values[0]);
      // AVX512DFTc2c(xt, xt + 1, xf, xf + 1, 2, 2, batch_size, (2 * fft_size),
      //             (2 * fft_size), fft_size);

      // avx512_c2c_scatter(fft_size, batch_size, xf, &out_array[0]);

      for (int i = 0; i < 2 * fft_size * batch_size; i += 2) {
        if (std::abs(arg1[0][i / 2][0] - xf_fftw[i / 2][0]) > 1e-3 ||
            std::abs(arg1[1][i / 2][0] - xf_fftw[i / 2][1]) > 1e-3) {
          std::cerr << "ours[" << i / 2 << "]\t Real: " << arg1[0][i / 2][0]
                    << ", complex: " << arg1[1][i / 2][0] << "\n";
          std::cerr << "FFTW[" << i / 2 << "]\t Real: " << xf_fftw[i / 2][0]
                    << ", complex: " << xf_fftw[i / 2][1] << "\n\n";
        }
      }

      fftw_free(xt_fftw);
      fftw_free(xf_fftw);
    }
}
*/
TEST(TestDFT, correctness_fftc) {
  // c2c
  int fft_size_init[7] = {64, 128, 256, 512, 1024, 2048, 4096};
  int batch_size_init[1] = {1};
  // double arg0[256][2][1];
  // double arg1[256][2][1];
  for (size_t i = 0; i < sizeof(fft_size_init) / sizeof(fft_size_init[0]); i++)
    for (size_t j = 0; j < sizeof(batch_size_init) / sizeof(batch_size_init[0]);
         j++) {
      int fft_size = fft_size_init[i];
      int batch_size = batch_size_init[j];
      double *arg0 = nullptr;
      double *arg1 = nullptr;
      size_t byte_size = 2 * fft_size * batch_size * sizeof(double);

      ::posix_memalign((void **)(&arg0), 64, byte_size);
      ::posix_memalign((void **)(&arg1), 64, byte_size);

      std::random_device rd;
      std::mt19937 generator(rd());
      std::uniform_real_distribution<double> dist(-10.0, 10.0);
      // Store values in vector
      std::vector<double> values(2 * fft_size * batch_size);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = dist(generator);
        //   values[i] = 1;
      }
      fftw_complex *xt_fftw = fftw_alloc_complex(fft_size * batch_size);
      fftw_complex *xf_fftw = fftw_alloc_complex(fft_size * batch_size);
      // fftw_complex *xt_fftc = fftw_alloc_complex(fft_size * batch_size);
      // fftw_complex *xf_fftc = fftw_alloc_complex(fft_size * batch_size);

      fftw_plan plan = fftw_plan_many_dft(
          1, &fft_size, batch_size, xt_fftw, &fft_size, 1, fft_size, xf_fftw,
          &fft_size, 1, fft_size, FFTW_FORWARD, FFTW_MEASURE);

      for (size_t i = 0; i < values.size(); i += 2) {
        xt_fftw[i / 2][0] = values[i];
        xt_fftw[i / 2][1] = values[i + 1];
      }

      fftw_execute(plan);

      /* for (int i = 0; i < 256; i++) {
         for (int j = 0; j < 2; j++) {
           for (int k = 0; k < 1; k++) {
             arg0[i][j][k] = values[i * 2 + j + k];
           }
         }
       }*/
      for (size_t i = 0; i < values.size(); i += 1) {
        arg0[i] = values[i];
      }
      FFTcDFTc2c(arg0, arg0 + 1, arg1, arg1 + 1, 2, 2, 1, 1, 1, fft_size);


      // std::vector<double> out_array(2 * fft_size * batch_size);
      // avx512_c2c_gather(fft_size, batch_size, xt, &values[0]);
      // AVX512DFTc2c(xt, xt + 1, xf, xf + 1, 2, 2, batch_size, (2 * fft_size),
      //             (2 * fft_size), fft_size);

      // avx512_c2c_scatter(fft_size, batch_size, xf, &out_array[0]);

      for (int i = 0; i < 2 * fft_size * batch_size; i += 2) {
        if (std::abs(arg1[i] - xf_fftw[i / 2][0]) > 1e-3 ||
            std::abs(arg1[i + 1] - xf_fftw[i / 2][1]) > 1e-3) {
          std::cerr << "ours[" << i / 2 << "]\t Real: " << arg1[i]
                    << ", complex: " << arg1[i + 1] << "\n";
          std::cerr << "FFTW[" << i / 2 << "]\t Real: " << xf_fftw[i / 2][0]
                    << ", complex: " << xf_fftw[i / 2][1] << "\n\n";
        }
      }

      fftw_free(xt_fftw);
      fftw_free(xf_fftw);
    }
}

TEST(TestDFT, FFTcPerfc2c) {
  double elaps[Rounds];
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int num = nsamp * 2;
  int ompT = 1;

  for (int k = 0; k < Rounds; k++) {

    // for (int i = 0; i < Iter; i++) {
    double *xt = nullptr;
    double *xf = nullptr;

    ::posix_memalign((void **)&xt, 64, nloop * 2 * nsamp * sizeof(double));
    ::posix_memalign((void **)&xf, 64, nloop * 2 * nsamp * sizeof(double));
    // fftw_complex *xt = fftw_alloc_complex(nsamp * nloop);
    // fftw_complex *xf = fftw_alloc_complex(nsamp * nloop);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    // Store values in vector
    std::vector<double> values(2 * nsamp * nloop);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = dist(generator);
    }
    for (size_t i = 0; i < values.size(); i += 1) {
      xt[i] = values[i];
    }

    using namespace std::chrono;
    high_resolution_clock::time_point iStart = high_resolution_clock::now();

    for (int i = 0; i < Iter; i++)
      FFTcDFTc2c(xt, xt + 1, xf, xf + 1, 2, 2, nloop, num, num, nsamp);
    // fft_memref_16_256_inter_cpu_novec((double *)xt, (double *)xt, 0, L, M, N,
    // L, M, 1,
    //                (double *)xf, (double *)xf, 0, L, M, N, L, M, 1);

    high_resolution_clock::time_point afterKernel =
        high_resolution_clock::now();

    duration<double, std::milli> kernel = afterKernel - iStart;
    elaps[k] = kernel.count();
    ::free(xf);
    ::free(xt);
  }

  for (int k = 0; k < Rounds; k++)
    std::cout << "round " << k << ": " << elaps[k] << std::endl;

  double sd = calculateSD(elaps);
  double mean = calculateMean(elaps);

  std::cout << "FFTc_Perf : nsamp : " << nsamp << " nloop : " << nloop
            << " ompT : " << ompT << " kernel mean : " << mean << " sd : " << sd
            << " .\n ";
}

TEST(TestDFT, correctness_fftc_2_4) {
  // c2c
  int fft_size_init[1] = {4};
  int batch_size_init[1] = {1};
  double arg0[2][4][1];
  double arg1[2][4][1];
  for (size_t i = 0; i < sizeof(fft_size_init) / sizeof(fft_size_init[0]); i++)
    for (size_t j = 0; j < sizeof(batch_size_init) / sizeof(batch_size_init[0]);
         j++) {
      int fft_size = fft_size_init[i];
      int batch_size = batch_size_init[j];
      // __m512 *xt = nullptr;
      //__m512 *xf = nullptr;
      // size_t byte_size = 2 * fft_size * batch_size * sizeof(double);

      // ::posix_memalign((void **)(&xt), 64, byte_size);
      // ::posix_memalign((void **)(&xf), 64, byte_size);

      std::random_device rd;
      std::mt19937 generator(rd());
      std::uniform_real_distribution<double> dist(-10.0, 10.0);
      // Store values in vector
      std::vector<double> values(2 * fft_size * batch_size);
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = dist(generator);
        //       values[i] = i;
      }
      fftw_complex *xt_fftw = fftw_alloc_complex(fft_size * batch_size);
      fftw_complex *xf_fftw = fftw_alloc_complex(fft_size * batch_size);
      // fftw_complex *xt_fftc = fftw_alloc_complex(fft_size * batch_size);
      // fftw_complex *xf_fftc = fftw_alloc_complex(fft_size * batch_size);

      fftw_plan plan = fftw_plan_many_dft(
          1, &fft_size, batch_size, xt_fftw, &fft_size, 1, fft_size, xf_fftw,
          &fft_size, 1, fft_size, FFTW_FORWARD, FFTW_MEASURE);

      for (size_t i = 0; i < values.size(); i += 2) {
        xt_fftw[i / 2][0] = values[i];
        xt_fftw[i / 2][1] = values[i + 1];
      }

      fftw_execute(plan);

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
          for (int k = 0; k < 1; k++) {

            arg0[i][j][k] = values[k * 8 + 2 * j + i];
            arg1[i][j][k] = 0;
          }
        }
      }
      fft_memref_2_4((double *)arg0, (double *)arg0, 0, 2, 46, 15, 44, 13, 14,
                     (double *)arg1, (double *)arg1, 0, 22, 43, 14, 45, 16, 1);

      // std::vector<double> out_array(2 * fft_size * batch_size);
      // avx512_c2c_gather(fft_size, batch_size, xt, &values[0]);
      // AVX512DFTc2c(xt, xt + 1, xf, xf + 1, 2, 2, batch_size, (2 * fft_size),
      //             (2 * fft_size), fft_size);

      // avx512_c2c_scatter(fft_size, batch_size, xf, &out_array[0]);

      for (int i = 0; i < 2 * fft_size * batch_size; i += 2) {
        if (std::abs(arg1[0][i / 2][0] - xf_fftw[i / 2][0]) > 1e-3 ||
            std::abs(arg1[1][i / 2][0] - xf_fftw[i / 2][1]) > 1e-3) {
          std::cerr << "ours[" << i / 2 << "]\t Real: " << arg1[0][i / 2][0]
                    << ", complex: " << arg1[1][i / 2][0] << "\n";
          std::cerr << "FFTW[" << i / 2 << "]\t Real: " << xf_fftw[i / 2][0]
                    << ", complex: " << xf_fftw[i / 2][1] << "\n\n";
        }
      }

      fftw_free(xt_fftw);
      fftw_free(xf_fftw);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
