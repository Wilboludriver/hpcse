#include <omp.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <array>

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

    // Create Parallel Region
    #pragma omp parallel reduction(+ : pi)
    {
        // Get threadId and seed random number generator
        int threadId = omp_get_thread_num();
        std::default_random_engine g(threadId);

        // Perform summation in parallel
        #pragma omp for
        for (size_t i = 0; i < N; ++i) {
            std::uniform_real_distribution<double> u;
            double x = u(g);
            double y = u(g);
            pi += F(x, y);
        }
    }
    return 4 * pi / N;
}

// Method 2: parallelize C0 only 
// using `omp parallel for reduction`, use arrays without padding
double C2(size_t N)
{
    // Get the maximum number of threads
    size_t maxNumThreads = omp_get_max_threads();

    // Create and seed one generator for each thread
    std::vector<std::default_random_engine> generators(maxNumThreads);
    for (size_t t = 0; t < maxNumThreads; ++t) {
        // Avoid using 0 as seed
        generators[t].seed(t + 1);
    }

    double pi = 0.;

    // Perform parallel reduction
    #pragma omp parallel for reduction(+ : pi)
    for (size_t i = 0; i < N; ++i) {
        size_t threadId = omp_get_thread_num();
        std::uniform_real_distribution<double> u;
        double x = u(generators[threadId]);
        double y = u(generators[threadId]);
        pi += F(x, y);
    }

    return 4 * pi / N;
}

// Method 3: parallelize C0 only
// using `omp parallel for reduction`, use arrays with padding
double C3(size_t N)
{
    // Struct with generator and padding
    // Make sure that the padding matches cache line size (https://stackoverflow.com/questions/794632/programmatically-get-the-cache-line-size)
    struct paddedRNG { 
        std::default_random_engine generator;
        char padding[64];
    };

    // Get the maximum number of threads
    size_t maxNumThreads = omp_get_max_threads();

    // Create and seed one generator for each thread
    std::vector<paddedRNG> paddedGenerators(maxNumThreads);
    for (size_t t = 0; t < maxNumThreads; ++t) {
        // Avoid using 0 as seed
        paddedGenerators[t].generator.seed(t + 1);
    }

    double pi = 0.;

    // Perform parallel reduction
    #pragma omp parallel for reduction(+ : pi)
    for (size_t i = 0; i < N; ++i) {
        size_t threadId = omp_get_thread_num();
        std::uniform_real_distribution<double> u;
        double x = u(paddedGenerators[threadId].generator);
        double y = u(paddedGenerators[threadId].generator);
        pi += F(x, y);
    }

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
