#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

char flushCache()
{
    constexpr std::size_t n = 1 << 20;
    static volatile char cacheFiller[n];
    for (std::size_t i = 0; i < n; ++i)
        cacheFiller[i] = 0;
    return cacheFiller[0];
}

void Ax_row(const std::vector<double>& A, const std::vector<double>& x,
            std::vector<double>& y)
{
    const size_t N = x.size();
    y.assign(N, 0.0);

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            y[i] += A[i * N + j] * x[j];
}

void Ax_col(const std::vector<double>& A, const std::vector<double>& x,
            std::vector<double>& y)
{
    const size_t N = x.size();
    y.assign(N, 0.0);

    for (size_t j = 0; j < N; j++)
        for (size_t i = 0; i < N; i++)
            y[i] += A[j * N + i] * x[j];
}

double benchmark_Ax(const std::vector<double>& A, const std::vector<double>& x,
                    bool row_major, double Ns)
{
    double times = 0;
    // preallocate buffers
    std::vector<double> y(x.size());

    for (size_t i = 0; i < Ns; i++) {
        flushCache();
        auto t1 = std::chrono::system_clock::now();
        // TOTO: Question 4a: Call the function to be benchmarked
        if (row_major == true)
            Ax_row(A, x, y);
        else
            Ax_col(A, x, y);
        auto t2 = std::chrono::system_clock::now();
        times += std::chrono::duration<double>(t2 - t1).count();
    }
    printf("Done in total %9.4fs  --  average %9.4fs\n", times, times / Ns);

    return times / Ns;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s [N|matrix dimension] [Ns|number of iterations]\n",
               argv[0]);
        exit(0);
    }

    size_t N = atoi(argv[1]);
    size_t Ns = atoi(argv[2]);
    std::vector<double> A(N * N), B(N * N), x(N);

    // TODO: Question 4a: Initialize matrices and vector
    //       store A as row major and B as column major
    for (size_t i = 0; i < N; i++) {
        x[i] = i;
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = 2 * i + j;
            B[i + j * N] = 2 * i + j;
        }
    }
    printf("Working with matrix of dimension %zu\n", N);

    printf("A*x (row major).\n");
    const double times1 = benchmark_Ax(A, x, true, Ns);

    printf("A*x (column major).\n");
    const double times2 = benchmark_Ax(B, x, false, Ns);

    printf("-----------------\n");
    printf("T_row / T_col = %.8f\n", times1 / times2);

    return 0;
}
