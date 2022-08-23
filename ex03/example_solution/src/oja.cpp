#include "oja.hpp"
#include <cblas.h>

void OjasRule::backward(const scalar* data, int batch_size) {
    
    // Compute outputs_ij = data_i * W_j
    forward(data, outputs_, batch_size);

    // Fill grad_ with weights_
    std::copy(weights_, weights_ + in_dims_, grad_);

    // Compute norm of outputs_
    scalar norm = cblas_dnrm2(batch_size, outputs_, 1);

    // Compte grad_ = (X^T * X - ||X * W||^2 * I) * W
    cblas_dgemv(CblasRowMajor, CblasTrans,
                batch_size, in_dims_,
                1.0, data, in_dims_, outputs_, 1,
                -norm * norm, grad_, 1
    );

}
