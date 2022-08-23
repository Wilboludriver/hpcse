#include <cassert>
#include <chrono>
#include <limits>
#include <math.h>
#include <string>
#include <vector>

void flush_cache()
{
    static volatile char buf[(1 << 20) * 64];
    for (size_t i = 0; i < sizeof(buf); ++i) {
        buf[i] += i;
    }
}

void transpose(std::vector<double>& A, std::vector<double>& AT)
{
    size_t N = sqrt(A.size());

// TODO: Question 4b: Straightforward matrix transposition
#ifdef WRITESTRIDE
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
#else
    for (size_t j = 0; j < N; j++)
        for (size_t i = 0; i < N; i++)
#endif
            AT[j * N + i] = A[i * N + j];
}

void transpose_block(std::vector<double>& A, std::vector<double>& AT,
                     size_t blockSize)
{
    size_t N = sqrt(A.size());

    // TODO: Question 4b: Block matrix transposition
    for (size_t j = 0; j < N; j += blockSize)
        for (size_t i = 0; i < N; i += blockSize)
#ifdef WRITESTRIDE
            for (size_t k = i; k < i + blockSize; k++)
                for (size_t l = j; l < j + blockSize; l++)
#else
            for (size_t l = j; l < j + blockSize; l++)
                for (size_t k = i; k < i + blockSize; k++)
#endif
                    AT[k + l * N] = A[l + k * N];
}

double benchmark_transpose(std::vector<double>& A, size_t mode,
                           size_t blockSize, size_t Ns, size_t batch)
{
    size_t N = sqrt(A.size());

    // TODO: Check that the matrix size is divided by the blockSize when mode==2
    if (mode == 2 && N % blockSize != 0) {
        fprintf(stderr,
                "Error: the size of the matrix (%zu) should be divided by the "
                "blockSize variable (%zu).\n",
                N, blockSize);
        exit(1);
    }

    std::vector<double> AT(N * N);
    double mintime = std::numeric_limits<double>::max();
    double maxtime = 0;
    double avgtime = 0;

    for (size_t i = 0; i < Ns; i++) {
        flush_cache();
        auto t1 = std::chrono::system_clock::now();
        // TODO: Question 4b: Call the function to be benchmarked
        for (size_t k = 0; k < batch; ++k) {
            if (mode == 1)
                transpose(A, AT);
            else if (mode == 2)
                transpose_block(A, AT, blockSize);
            std::swap(A, AT);
        }
        auto t2 = std::chrono::system_clock::now();
        const auto dt = std::chrono::duration<double>(t2 - t1).count();
        mintime = std::min(mintime, dt);
        maxtime = std::max(maxtime, dt);
        avgtime += dt;
    }
    avgtime /= Ns;
    fprintf(stderr,
            "Done, min=%9.4f, avg=%9.4f, max=%9.4f, batch=%zu, Ns=%zu\n",
            mintime, avgtime, maxtime, batch, Ns);

    return mintime / batch;
}

int main()
{
    std::vector<size_t> matrixSize;
    for (int m = 512; m <= 4096; m *= 2) {
        matrixSize.push_back(m);
    }
    size_t M = matrixSize.size();

    std::vector<size_t> blockSize;
    for (int b = 2; b <= 512; b *= 2) {
        blockSize.push_back(b);
    }
    size_t B = blockSize.size();

    const size_t Ns = 10; // number of samples in benchmark

    std::vector<double> times1(M, 0);
    std::vector<std::vector<double>> times2(B, std::vector<double>(M, 0));

    std::vector<double> A;

    // loop over matrix sizes
    for (size_t m = 0; m < M; m++) {
        fprintf(stderr, "Working with a matrix of size %zu\n", matrixSize[m]);

        const size_t N = matrixSize[m];
        const size_t N0 = 4096; // size for which batch=1 is representative
        const size_t batch = std::max<size_t>(1, N0 * N0 / (N * N));

        // TODO: Initialize the matrix
        A.resize(N * N);
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                A[i * N + j] = 2 * i + j;

        fprintf(stderr, "Start transposing (non optimized).\n");
        times1[m] = benchmark_transpose(A, 1, 0, Ns, batch);

        // loop over block sizes
        for (size_t b = 0; b < B; b++) {
            if (matrixSize[m] < blockSize[b]) {
                continue;
            }
            fprintf(stderr, "Start transposing (optimized, block size=%zu).\n",
                    blockSize[b]);
            times2[b][m] = benchmark_transpose(A, 2, blockSize[b], Ns, batch);
        }

        fprintf(stderr, "==================================================\n");
    }

    // write results to a file
    FILE* fp = nullptr;
    fp = fopen("transpose_times.txt", "w");
    // write header to the file
    std::string header = " N   time_unoptimized ";
    for (size_t b = 0; b < B; b++)
        header = header + "  block_" + std::to_string(blockSize[b]);
    header = header + "\n";
    fprintf(fp, "%s", header.c_str());
    for (size_t m = 0; m < M; m++) {
        fprintf(fp, "%zu %lg", matrixSize[m], times1[m]);
        for (size_t b = 0; b < B; b++)
            fprintf(fp, " %lg ", times2[b][m]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}
