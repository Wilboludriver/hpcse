#include "sanger.hpp"
#include <cblas.h>

SangersRule::SangersRule(int in_dims, int out_dims, int max_batch)
  :  inference_(new scalar[out_dims_]), Perceptron(in_dims, out_dims, max_batch)
{}

SangersRule::~SangersRule() {
    delete[] inference_;
}

void SangersRule::backward(const scalar* data, int batch_size) {
    
    // Compute outputs and save in buffer outputs_
    forward(data, outputs_, batch_size);

    // Compte first component of gradient similar to Hebb's rule
    // The fist component is: X^T * X * W = X^T * O
    // Save intermediate results to grad_
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                in_dims_, out_dims_, batch_size,
                1.0, data, in_dims_, outputs_, out_dims_,
                0.0, grad_, out_dims_
    );

    // Iterate over output dimensions
    for(int k = 0; k < out_dims_; ++k) {

        // Compute inference I_k = [o_1^T ... o_k^T] * o_k
        // Save inference in inference_
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    batch_size, k + 1,
                    1.0, outputs_, out_dims_, outputs_ + k, out_dims_,
                    0.0, inference_, 1
        );

        // Refine the gradient using the inference
        // Subtract from the k-th column of grad_: [w_1 ... w_k] * I_k
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    in_dims_, k + 1,
                    -1.0, weights_, out_dims_, inference_, 1,
                    1.0, grad_ + k, out_dims_
        );

    }

}
