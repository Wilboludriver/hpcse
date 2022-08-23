#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "utils/dataloader.hpp"
#include "utils/matrix2file.hpp"

#ifdef _MKL_
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

// NUM_POINTS controls how many points are loaded
// For debugging or developing on your laptop with limited RAM you
// can decrease this value. You should never have to lower the
// value below 100. When you have debugged your code compile it once
// with "make config=prod" and run it to observe how the time scales
// with the problem size.

#ifndef NDEBUG
#define NUM_POINTS 3000
#else
#define NUM_POINTS 15000
#endif

int main(int argc, char *argv[]) {

  // Load data from files. See utils/dataloader.hpp for more information.
  utils::dataloader loader(NUM_POINTS); // Create dataloader
  loader.load();                        // load data
  loader.compute_delta();               // compute deltas

  // Allocate memory
  int N = NUM_POINTS - 2;
  double *A = new double[N * N]();
  double *A_sym = new double[N * N]();
  double *b_dense = new double[N]();
  double *b_symmetric = new double[N]();
  double *b_tridiagonal = new double[N]();
  // Arrays for routines
  int *ipiv = new int[N]();
  int info = 0;

  // Initialize timers
  auto timer = std::chrono::high_resolution_clock();
  auto tstart = timer.now();
  auto tend = timer.now();
  std::chrono::duration<double> time;

  // Initialize matrices

  // Dense and symmetric
  for (int i = 0; i < N; i++) {
    A[i + N * i] = (loader.delta_x[i] + loader.delta_x[i + 1]) / 3;
  }

  for (int i = 0; i < N - 1; i++) {
    A[i + 1 + N * i] = loader.delta_x[i + 1] / 6;
  }

  for (int i = 0; i < N - 1; i++) {
    A[i + N * (i + 1)] = loader.delta_x[i + 1] / 6;
  }

  std::copy_n(A, N * N, A_sym);

  // Tridiagonal
  // The tridiagonal routines might need the matrix
  // A in a different storage layout. Refer to the
  // memory allocation above
  // IMPORTANT: You have to deallocate all memory
  // that you allocate!
  double *A_diag_major = new double[N]();
  double *A_diag_upper = new double[N - 1]();
  double *A_diag_lower = new double[N - 1]();

  for (int i = 0; i < N; i++) {
    A_diag_major[i] = (loader.delta_x[i] + loader.delta_x[i + 1]) / 3;
  }

  for (int i = 0; i < N - 1; i++) {
    A_diag_upper[i] = loader.delta_x[i + 1] / 6;
  }

  std::copy_n(A_diag_upper, N - 1, A_diag_lower);

  // Initialize rhs vector b
  for (int i = 0; i < N; i++) {
    b_dense[i] = loader.delta_y[i + 1] / loader.delta_x[i + 1] -
                 loader.delta_y[i] / loader.delta_x[i];
  }

  std::copy_n(b_dense, N, b_symmetric);
  std::copy_n(b_dense, N, b_tridiagonal);

  std::cout << "Initialization complete!" << std::endl;

  // Dense routine
  tstart = timer.now();

  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, 1, A, N, ipiv, b_dense, 1);

  tend = timer.now();

// Checking for errors
#ifndef NDEBUG
  if (info != 0) {
    throw std::runtime_error("Info != 0 for dense routine! Info = " +
                             std::to_string(info));
  }
#endif

  time = (tend - tstart);

  std::cout << "Dense routine needed: " << time.count() << "s" << std::endl;

  // Symmetric routine
  tstart = timer.now();

  info = LAPACKE_dsysv(LAPACK_ROW_MAJOR, 'U', N, 1, A_sym, N, ipiv, b_symmetric,
                       1);

  tend = timer.now();

// Checking for errors
#ifndef NDEBUG
  if (info != 0) {
    throw std::runtime_error("Info != 0 for symmetric routine! Info = " +
                             std::to_string(info));
  }
#endif

  time = (tend - tstart);

  std::cout << "Symmetrical routine needed: " << time.count() << "s"
            << std::endl;

  // Tridiagonal routine
  tstart = timer.now();

  info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, N, 1, A_diag_lower, A_diag_major,
                       A_diag_upper, b_tridiagonal, 1);

  tend = timer.now();

// Checking for errors
#ifndef NDEBUG
  if (info != 0) {
    throw std::runtime_error("Info != 0  for diagonal routine! Info = " +
                             std::to_string(info));
  }
#endif

  time = (tend - tstart);

  std::cout << "Tridiagonal routine needed: " << time.count() << "s"
            << std::endl;

  utils::matrix2file(b_dense, "b_dense", 1, N);
  utils::matrix2file(b_symmetric, "b_symmetric", 1, N);
  utils::matrix2file(b_tridiagonal, "b_tridiagonal", 1, N);

  // Free allocated memory
  delete[] A;
  delete[] A_sym;
  delete[] A_diag_major;
  delete[] A_diag_upper;
  delete[] A_diag_lower;
  delete[] b_dense;
  delete[] b_symmetric;
  delete[] b_tridiagonal;
  delete[] ipiv;

  return 0;
}