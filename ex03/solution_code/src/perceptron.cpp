#include "perceptron.hpp"
#include <cblas.h>
#include <cmath>

Perceptron::Perceptron(int in_dims, int out_dims, int max_batch)
  : in_dims_(in_dims),
    out_dims_(out_dims),
    max_batch_(max_batch),
    weights_(new scalar[in_dims * out_dims]),
    grad_(new scalar[in_dims * out_dims]),
    outputs_(new scalar[max_batch * out_dims_])
{

    // Clear gradient to avoid problems with NaN and Inf
    // We initialize weights_ with 1 / in_dims_ 
    // as 0 would have lead to no updating of the weights
    std::fill(weights_, weights_ + in_dims * out_dims , 1.0 / in_dims_);
    std::fill(grad_, grad_ + in_dims * out_dims , 0.0);
    std::fill(outputs_, outputs_ + max_batch * out_dims_, 0.0);

}

Perceptron::~Perceptron() {

    // Clean up
    delete[] weights_;
    delete[] grad_;
    delete[] outputs_;

}

void Perceptron::forward(const scalar* data, scalar* out, const int batch_size) const {

    // Compute O = X * W
    // out_ij = data_i: * weights_:j
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, out_dims_, in_dims_,
                1.0, data, in_dims_, weights_, out_dims_,
                0.0, out, out_dims_
    );

}

void Perceptron::normalize_grad() {

    // Normalize the gradient of each weight vector individually
    for(int k = 0; k < out_dims_; ++k) {
        const scalar norm = cblas_dnrm2(in_dims_, grad_ + k, out_dims_);
        cblas_dscal(in_dims_, 1 / norm, grad_ + k, out_dims_);
    }

}

void Perceptron::update_weights(const scalar* data, int batch_size, const scalar lr) {

    // Compute gradient
    backward(data, batch_size);

    // Normalize gradient
    normalize_grad();

    // Compute weights += lr * grad
    cblas_daxpy(in_dims_ * out_dims_, lr, grad_, 1, weights_, 1);

}

void Perceptron::get_spectrum(const scalar* data, int batch_size, scalar* eigen_values, scalar* eigen_vectors) const {

    // Compute outputs_ij = data_i * W_j
    forward(data, outputs_, batch_size);

    // Compute 
    for(int k = 0; k < out_dims_; ++k) {

        // Compute eigen value as || X * w_k ||^2 / (batch_size - 1)
        if(eigen_values !=  nullptr) {
            const scalar norm = cblas_dnrm2(batch_size, outputs_ + k, out_dims_);
            eigen_values[k] = norm * norm / (batch_size - 1); 
        }
        

        // Eigen vector are the columns of weights_
        // However in eigen_vectors it should be saves as a row vector
        if(eigen_vectors !=  nullptr)
            cblas_dcopy(in_dims_, weights_ + k, out_dims_, eigen_vectors + k * in_dims_, 1);
        
    }

}

int Perceptron::get_in_dims() const {
    return in_dims_;
}

int Perceptron::get_out_dims() const {
    return out_dims_;
}

int Perceptron::get_max_batch() const {
    return max_batch_;
}
