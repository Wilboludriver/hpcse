#include "hebb.hpp"
#include <cblas.h>

void HebbsRule::backward(const scalar* data, int batch_size) {
    
    // Compute outputs = X * W
    forward(data, outputs_, batch_size);

    // Compte grad = X^T * X * W = X^T * outputs
    cblas_dgemv(CblasRowMajor, CblasTrans,
                batch_size, in_dims_,
                1.0, data, in_dims_, outputs_, 1,
                0.0, grad_, 1
    );

}
