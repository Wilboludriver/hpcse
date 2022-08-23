#include "gemv_avx2.h"
#include <assert.h>
#include <chrono>
#include <cstring>
#include <immintrin.h> // Provides intrinsics
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdlib.h> // Provides posix_memalign
#include <string>

const size_t SIMD_WIDTH = 256;
const size_t SIMD_WIDTH_BYTES = SIMD_WIDTH / 8;
const size_t SIMD_WIDTH_FLOATS = SIMD_WIDTH_BYTES / sizeof(float);

const size_t SIZE = 1000;

void init_array(float *Array, size_t N, size_t M) {
  std::mt19937 gen(1);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      Array[i * N + j] = dist(gen);
    }
  }
}

// naive implementation
void sgemv(const float *A, const float *x, float *y, size_t N) {

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      y[i] += A[i * N + j] * x[j];
    }
  }
}

// Using AVX instructions
void sgemv_simd(const float *A, const float *x, float *y, size_t N) {

  size_t i, j;
  __m256 AA, xx, yy;

  for (i = 0; i < N; ++i) {
    yy = _mm256_set_ps(0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

    for (j = 0; j < N; j += SIMD_WIDTH_FLOATS) {
      AA = _mm256_load_ps(A + i * N + j);
      xx = _mm256_load_ps(x + j);
      yy = _mm256_fmadd_ps(AA, xx, yy);
    }

    y[i] = ((yy[0] + yy[1]) + (yy[2] + yy[3])) +
           ((yy[4] + yy[5]) + (yy[6] + yy[7]));
  }
}

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  assert(SIZE % SIMD_WIDTH_FLOATS == 0);

  // Initialize arrays
  float *A, *x, *y, *y_simd, *y_ispc;

  posix_memalign((void **)&A, SIMD_WIDTH_BYTES, SIZE * SIZE * sizeof(*A));
  posix_memalign((void **)&x, SIMD_WIDTH_BYTES, SIZE * sizeof(*x));
  posix_memalign((void **)&y, SIMD_WIDTH_BYTES, SIZE * sizeof(*y));
  posix_memalign((void **)&y_simd, SIMD_WIDTH_BYTES, SIZE * sizeof(*y_simd));
  posix_memalign((void **)&y_ispc, SIMD_WIDTH_BYTES, SIZE * sizeof(*y_ispc));

  init_array(A, SIZE, SIZE);
  init_array(x, SIZE, 1);
  memset(y, 0, SIZE * sizeof(*y));
  memset(y_simd, 0, SIZE * sizeof(*y_simd));
  memset(y_simd, 0, SIZE * sizeof(*y_simd));

  // Initialize timer
  auto timer = std::chrono::high_resolution_clock();
  auto tstart = timer.now();
  auto tend = timer.now();
  std::chrono::duration<float> time_naive;
  std::chrono::duration<float> time_simd;
  std::chrono::duration<float> time_ispc;

  // Test naive implementation
  tstart = timer.now();
  sgemv(A, x, y, SIZE);
  tend = timer.now();

  time_naive = tend - tstart;

  // Test SIMD implementation
  tstart = timer.now();
  sgemv_simd(A, x, y_simd, SIZE);
  tend = timer.now();

  time_simd = tend - tstart;

  // Test ISPC implementation
  tstart = timer.now();
  ispc::sgemv_ispc(A, x, y_ispc, SIZE);
  tend = timer.now();

  time_ispc = tend - tstart;

#ifndef NDEBUG
  std::cout << "Checking for correctness of SIMD implementation..."
            << std::endl;
  for (size_t i = 0; i < SIZE; i++) {
    if (std::abs(y[i] - y_simd[i]) > 1e-3)
      throw(std::runtime_error(
          "Error detected: Y naive: " + std::to_string(y[i]) +
          "  Y SIMD: " + std::to_string(y_simd[i])));
  }
  std::cout << "SIMD result is correct!" << std::endl;

  std::cout << "Checking for correctness of ISPC implementation..."
            << std::endl;
  for (size_t i = 0; i < SIZE; i++) {
    if (std::abs(y[i] - y_ispc[i]) > 1e-3)
      throw(std::runtime_error(
          "Error detected: Y naive: " + std::to_string(y[i]) +
          "  Y ISPC: " + std::to_string(y_simd[i])));
  }
  std::cout << "ISPC result is correct!" << std::endl;
#endif

  std::cout << "Naive implementation used: " << time_naive.count() << "s"
            << std::endl;
  std::cout << "SIMD implementation used:  " << time_simd.count()
            << "s Speedup: " << time_naive.count() / time_simd.count()
            << std::endl;
  std::cout << "ISPC implementation used:  " << time_ispc.count()
            << "s Speedup: " << time_naive.count() / time_ispc.count()
            << std::endl;

  free(A);
  free(x);
  free(y);
  free(y_simd);
  free(y_ispc);
}
