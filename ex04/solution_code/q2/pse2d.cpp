#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>

size_t SQRT_N;                      // Number of particles per dimension.
size_t N;                           // Number of particles.
constexpr double DOMAIN_SIZE = 2.0; // Domain side length.
constexpr double D = 1.0;           // Diffusion constant.
constexpr double dt = 0.0001;       // Time step.
constexpr double invPI = 1. / M_PI; // Inverse of Pi.

// TODO: 2c: Adjust parameter based on your estimate.
constexpr double eps = 0.03; // Epsilon.

// Particle storage.
std::vector<double> x;
std::vector<double> y;
std::vector<double> phi;
std::vector<double> dphi; // Temporary storage for the RHS.

// TODO 2b: Implement the Gaussian kernel.
double gaussian_eps(double dx, double dy) {
  constexpr double inveps = 1.0 / eps;
  constexpr double inveps2 = inveps * inveps;
  constexpr double factor = 4. * invPI * inveps2;

  const double dd = inveps2 * (dx * dx + dy * dy);
  return factor * std::exp(-dd);
}

/*
 * Initialize the particle positions and values.
 */
void init(void) {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.1, +0.1);

  for (size_t i = 0; i < SQRT_N; ++i)
    for (size_t j = 0; j < SQRT_N; ++j) {
      const size_t k = i * SQRT_N + j;

      /* Init partixles x coordinate */
      if (i == 0)
        x[k] = -0.5 * DOMAIN_SIZE; // Put some particles on lower x boundary
      else if (i == SQRT_N - 1)
        x[k] = 0.5 * DOMAIN_SIZE; // Put some particles on upper x boundary
      else
        x[k] = DOMAIN_SIZE * ((i + dis(gen)) / (SQRT_N - 1.) -
                              0.5); // Put particles on lattice with up to 10%
                                    // random displacement
      /* Init particles y coordinate */
      if (j == 0)
        y[k] = -0.5 * DOMAIN_SIZE; // Put some particles on lower y boundary
      else if (j == SQRT_N - 1)
        y[k] = 0.5 * DOMAIN_SIZE; // Put some particles on upper y boundary
      else
        y[k] = DOMAIN_SIZE * ((j + dis(gen)) / (SQRT_N - 1.) -
                              0.5); // Put particles on lattice with up to 10%
                                    // random displacement
      /* Set initial condition */
      if ((std::abs(x[k]) <= 0.25 * DOMAIN_SIZE) &&
          (std::abs(y[k]) <= 0.25 * DOMAIN_SIZE))
        phi[k] = 1.0;
    }
}

/*
 * Perform a single timestep. Compute RHS of Eq. (2) and update values of phi_i.
 */
// TODO  2d: Parallelize this function with OpenMP.
void timestep(void) {
  const double volume = DOMAIN_SIZE * DOMAIN_SIZE / N;
  const double factor = D * volume / (eps * eps);

  // TODO 2b: Implement the RHS.
#pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    double sum = 0;
    for (size_t j = i + 1; j < N; ++j) {
      const double tmp =
          (phi[j] - phi[i]) * gaussian_eps(x[i] - x[j], y[i] - y[j]);
      sum += tmp;
#pragma omp atomic
      dphi[j] -= factor * tmp;
    }
    dphi[i] += factor * sum;
  }

  // TODO 2b: Implement the forward Euler update of phi_i.
#pragma omp parallel for collapse(2)
  for (size_t i = 1; i < SQRT_N - 1;
       ++i) // skip updating of particles on x boundaries
    for (size_t j = 1; j < SQRT_N - 1;
         ++j) // skip updating of particles on y boundaries
    {
      const size_t k = i * SQRT_N + j;
      phi[k] += dt * dphi[k];
      dphi[k] = 0.; // reset dphi
    }
}

/*
 * Store the particles into a file.
 */
void save(int k) {
  char filename[32];
  sprintf(filename, "output/plot2d_%03d.txt", k);
  FILE *f = fopen(filename, "w");
  fprintf(f, "x  y  phi\n");
  for (size_t i = 0; i < N; ++i)
    fprintf(f, "%lf %lf %lf\n", x[i], y[i], phi[i]);
  fclose(f);
}

int main(int argc, char *argv[]) {

  if (argc < 4) {
    fprintf(stderr, "Usage: %s SQRT_N T out\n", argv[0]);
    return 1;
  }

  /* Init num particles */
  SQRT_N = std::stoul(argv[1]);
  N = SQRT_N * SQRT_N;

  /* Init simulation length */
  const double T = std::stod(argv[2]);

  /* Produce output */
  const int out = std::stoul(argv[3]); // YES == 1

  /* Init particle storage */
  x = std::vector<double>(N, 0.0);
  y = std::vector<double>(N, 0.0);
  phi = std::vector<double>(N, 0.0);
  dphi = std::vector<double>(N, 0.0);

  const size_t numSteps = T / (10. * dt);
  auto tstart = std::chrono::steady_clock::now();

  init();
  if (out == 1)
    save(0);
  for (size_t i = 1; i <= numSteps; ++i) {
    fprintf(stderr, ".");
    for (int j = 0; j < 10; ++j)
      timestep();
    if (out == 1)
      save(i); // Every 10 time steps == 1 frame of the animation.
  }
  fprintf(stderr, "\n");
  auto tend = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart)
          .count();
  printf("time: %lf\n", ms);

  return 0;
}
