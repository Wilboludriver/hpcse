#include <cassert>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

void AB(std::vector<double>& A, std::vector<double>& B, std::vector<double>& C)
{
    size_t N = sqrt(A.size());
    for (size_t i = 0; i < C.size(); i++)
        C[i] = 0;

    // TODO: Question 4c: Straightforward matrix-matrix multiplication
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void AB_block_row(std::vector<double>& A, std::vector<double>& B,
                  std::vector<double>& C, size_t blockSize)
{
    size_t N = sqrt(A.size());
    for (size_t i = 0; i < C.size(); i++)
        C[i] = 0;

    // TODO: Question 4c: Block matrix-matrix multiplication - B in row major
    for (size_t K = 0; K < N; K += blockSize)
        for (size_t I = 0; I < N; I += blockSize)
            for (size_t J = 0; J < N; J += blockSize)
                for (size_t i = I; i < I + blockSize; i++)
                    for (size_t j = J; j < J + blockSize; j++)
                        for (size_t k = K; k < K + blockSize; k++)
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
}

void AB_block_col(std::vector<double>& A, std::vector<double>& B,
                  std::vector<double>& C, size_t blockSize)
{
    size_t N = sqrt(A.size());
    for (size_t i = 0; i < C.size(); i++)
        C[i] = 0;

    // TODO: Question 4c: Block matrix-matrix multiplication - B in column major
    for (size_t K = 0; K < N; K += blockSize)
        for (size_t I = 0; I < N; I += blockSize)
            for (size_t J = 0; J < N; J += blockSize)
                for (size_t i = I; i < I + blockSize; i++)
                    for (size_t j = J; j < J + blockSize; j++)
                        for (size_t k = K; k < K + blockSize; k++)
                            C[i * N + j] += A[i * N + k] * B[k + N * j];
}

double benchmark_AB(std::vector<double>& A, std::vector<double>& B, size_t mode,
                    size_t blockSize, size_t Ns)
{
    size_t N = sqrt(A.size());
    double times = 0;

    // TODO: Check that the matrix size is divided by the blockSize when mode==2
    // or 3
    if ((mode == 2 or mode == 3) && N % blockSize != 0) {
        fprintf(stderr,
                "Error: the size of the matrix (%zu) should be divided by the "
                "blockSize variable (%zu).\n",
                N, blockSize);
        exit(1);
    }

    std::vector<double> C(N * N);

    for (size_t i = 0; i < Ns; i++) {
        auto t1 = std::chrono::system_clock::now();
        // TODO: Question 4c: Call the function to be benchmarked
        if (mode == 1)
            AB(A, B, C);
        else if (mode == 2)
            AB_block_row(A, B, C, blockSize);
        else if (mode == 3)
            AB_block_col(A, B, C, blockSize);
        auto t2 = std::chrono::system_clock::now();
        times += std::chrono::duration<double>(t2 - t1).count();
    }
    fprintf(stderr, "Done in total %9.4fs  --  average %9.4fs\n", times,
            times / Ns);

    return times / Ns;
}

int main()
{
    std::vector<size_t> matrixSize{512, 1024, 2048};
    size_t M = matrixSize.size();
    std::vector<size_t> Ns{10, 5, 1};
    assert(Ns.size() >= matrixSize.size());

    std::vector<size_t> blockSize;
    for (int b = 2; b <= 512; b *= 2) {
        blockSize.push_back(b);
    }
    size_t Bs = blockSize.size();

    std::vector<double> times1(M, 0);
    std::vector<std::vector<double>> times2(Bs, std::vector<double>(M, 0));
    std::vector<std::vector<double>> times3(Bs, std::vector<double>(M, 0));

    std::vector<double> A, B, C;

    for (size_t m = 0; m < M; m++) {

        fprintf(stderr, "Working with matrices of size %zu\n", matrixSize[m]);
        fprintf(stderr, "---------------------------------------------\n");

        size_t N = matrixSize[m];

        // TODO: Question 4c: Initialize matrices
        //       store A and B as row major and C as column major
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++) {
                A[i * N + j] = 1.1 * i + j;
                B[i * N + j] = 1.05 * i + j;
                C[i + N * j] = 1.05 * i + j;
            }

        fprintf(stderr, "Start C=A*B (non optimized).\n");
        times1[m] = benchmark_AB(A, B, 1, 0, Ns[m]);

        fprintf(stderr, "---------------------------------------------\n");

        for (size_t b = 0; b < Bs; b++) {
            fprintf(stderr,
                    "Start C=A*B (optimized, row major, block size=%zu).\n",
                    blockSize[b]);
            times2[b][m] = benchmark_AB(A, B, 2, blockSize[b], Ns[m]);
        }

        fprintf(stderr, "---------------------------------------------\n");

        for (size_t b = 0; b < Bs; b++) {
            fprintf(stderr,
                    "Start C=A*B (optimized, column major, block size=%zu).\n",
                    blockSize[b]);
            times3[b][m] = benchmark_AB(A, C, 3, blockSize[b], Ns[m]);
        }

        fprintf(stderr, "==================================================\n");
    }

    FILE* fp = nullptr;
    fp = fopen("matrix_matrix_times.txt", "w");
    // write header to the file
    std::string header = " N   unopt ";
    for (size_t b = 0; b < Bs; b++)
        header = header + "  br_" + std::to_string(blockSize[b]);
    for (size_t b = 0; b < Bs; b++)
        header = header + "  bc_" + std::to_string(blockSize[b]);
    header = header + "\n";
    fprintf(fp, "%s", header.c_str());

    for (size_t m = 0; m < M; m++) {
        fprintf(fp, "%zu %lf", matrixSize[m], times1[m]);
        for (size_t b = 0; b < Bs; b++)
            fprintf(fp, " %lf ", times2[b][m]);
        for (size_t b = 0; b < Bs; b++)
            fprintf(fp, " %lf ", times3[b][m]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}
