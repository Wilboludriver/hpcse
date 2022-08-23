#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mkl_lapacke.h>
#include <stdexcept>

#include "utils/dataloader.hpp"
#include "utils/matrix2file.hpp"

// NUM_POINTS controls how many points are loaded
// For debugging or developing on your laptop with limited RAM you
// can decrease this value. You should never have to lower the
// value below 100.
//#define NUM_POINTS 100
#define NUM_POINTS 15000

int main(int argc, char *argv[]) {

  // Load data from files
  utils::dataloader loader(NUM_POINTS);
  loader.load();
  loader.compute_delta();

  // Allocate memory
  // TODO: Set correct size for arrays
  int N = 1; // Set the correct value here
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
  // TODO: Initialize empty A matrices for dense and
  // symmetric routine

  // Tridiagonal
  // The tridiagonal routines might need the matrix
  // A in a different storage layout. Refer to the
  // memory allocation above
  // IMPORTANT: You have to deallocate all memory
  // that you allocate!
  // TODO: Initialize arrays for tridiagonal routine

  // Initialize rhs vector b
  // TODO: Initialize vector b (Hint std::copy_n might be useful)

  std::cout << "Initialization complete!" << std::endl;

  // Dense routine
  tstart = timer.now();
  // TODO: Call dense routine

  tend = timer.now();

  // Checking for errors
  if (info != 0) {
    throw std::runtime_error("Info != 0 for dense routine! Info = " +
                             std::to_string(info));
  }

  time = (tend - tstart);

  std::cout << "Dense routine needed: " << time.count() << "s" << std::endl;

  // Symmetric routine
  tstart = timer.now();
  // TODO: Call symmetric routine

  tend = timer.now();

  // Checking for errors
  if (info != 0) {
    throw std::runtime_error("Info != 0 for symmetric routine! Info = " +
                             std::to_string(info));
  }
  time = (tend - tstart);

  std::cout << "Symmetrical routine needed: " << time.count() << "s"
            << std::endl;

  // Tridiagonal routine
  tstart = timer.now();
  // TODO: Call tridiagonal routine

  tend = timer.now();

  // Checking for errors
  if (info != 0) {
    throw std::runtime_error("Info != 0  for diagonal routine! Info = " +
                             std::to_string(info));
  }

  time = (tend - tstart);

  std::cout << "Tridiagonal routine needed: " << time.count() << "s"
            << std::endl;

  // Write to disk
  utils::matrix2file(b_dense, "b_dense", 1, N);
  utils::matrix2file(b_symmetric, "b_symmetric", 1, N);
  utils::matrix2file(b_tridiagonal, "b_tridiagonal", 1, N);

  // Free allocated memory
  delete[] A;
  delete[] A_sym;
  delete[] b_dense;
  delete[] b_symmetric;
  delete[] b_tridiagonal;
  delete[] ipiv;

  return 0;
}