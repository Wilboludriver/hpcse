#include "sanger.hpp"
#include <cblas.h>

SangersRule::SangersRule(int in_dims, int out_dims, int max_batch)
  :  inference_(new scalar[out_dims_]), Perceptron(in_dims, out_dims, max_batch)
{}

SangersRule::~SangersRule() {
    delete[] inference_;
}

void SangersRule::backward(const scalar* data, int batch_size) {
    
    // TO DO: Compute outputs and save in buffer outputs_

    // TO DO: Compte first component of gradient similar to Hebb's rule
    // The fist component is: X^T * X * W = X^T * O
    // Save intermediate results to grad_

    // Iterate over output dimensions
    for(int k = 0; k < out_dims_; ++k) {

        // TO DO: Compute inference I_k = [o_1^T ... o_k^T] * o_k
        // Save inference in inference_

        // TO DO: Refine the gradient using the inference
        // Subtract from the k-th column of grad_: [w_1 ... w_k] * I_k

    }

}
