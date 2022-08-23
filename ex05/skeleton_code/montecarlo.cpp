#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

// Characteristic function for unit circle
inline double F(double x, double y)
{
    if (x * x + y * y < 1.) { // inside unit circle
        return 1.;
    }
    return 0.;
}

// Method 0: serial
double C0(size_t N)
{
    // random generator with seed 0
    std::default_random_engine g(0);

    // uniform distribution in [0, 1]
    std::uniform_real_distribution<double> u;

    // perform Monte-Carlo integration
    double pi = 0.;
    for (size_t i = 0; i < N; ++i) {
        double x = u(g);
        double y = u(g);
        pi += F(x, y);
    }
    return 4 * pi / N;
}

// Method 1: parallelize C0 without using arrays
double C1(size_t N)
{
    double pi = 0.;

    // TODO: Implement Method 1

    return 4 * pi / N;
}

// Method 2: parallelize C0 only using 
// `omp parallel for reduction`, use arrays without padding
double C2(size_t N)
{
    double pi = 0.;

    // TODO: Implement Method 2

    return 4 * pi / N;
}

// Method 2: parallelize C0 only using 
// `omp parallel for reduction`, use arrays without padding
double C3(size_t N)
{
    double pi = 0.;

    // TODO: Implement Method 3

    return 4 * pi / N;
}

// Returns integral of F(x,y) over unit square (0 < x < 1, 0 < y < 1).
// n: number of samples
// m: method
double C( size_t N, size_t method )
{
    switch ( method ) {
    case 0:
        return C0(N);
    case 1:
        return C1(N);
    case 2:
        return C2(N);
    case 3:
        return C3(N);
    default:
        printf("Unknown method '%ld'\n", method);
        abort();
    }
}

int main(int argc, char* argv[])
{
    // Default number of samples
    const size_t Ndef = 1e8;

    if (argc < 2 || argc > 3 || std::string(argv[1]) == "-h") {
        fprintf(stderr, "usage: %s METHOD [N=%ld]\n", argv[0], Ndef);
        fprintf(stderr, "Monte-Carlo integration with N samples.\n\
METHOD:\n\
0: serial\n\
1: openmp, no array\n\
2: `omp parallel for reduction`, array without padding\n\
3: `omp parallel for reduction`, array with padding\n");
        return 1;
    }

    // Parse method
    size_t method = atoi(argv[1]);

    // Parse number of samples
    size_t N = (argc > 2 ? atoi(argv[2]) : Ndef);

    // Reference solution
    double ref = M_PI;

    // Run and measure execution time
    double wt0 = omp_get_wtime();
    double res = C(N, method);
    double wt1 = omp_get_wtime();

    // Print result
    printf("res:  %.20f\nref:  %.20f\nerror: %.20e\ntime: %lf\n", res, ref,
           res - ref, wt1 - wt0);

    return 0;
}
