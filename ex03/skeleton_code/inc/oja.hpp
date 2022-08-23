#pragma once

#include "perceptron.hpp"

/**
 *  @short Implementation of Oja's Rule for computing
 *         eigen values and eigen vectors of a dataset's
 *         empirical covariance matrix.
 *
 *         Derives from Perceptron
 */
class OjasRule: public Perceptron {

public:

    /**
     * @short Constructor
     *
     * @param in_dims: Dimensionality of datapoints in the dataset
     * @param max_batch: Upper limit for batch size which will be
     *                   passed to forward, update_weights or get_spectrum  
     */
    OjasRule(int in_dims, int max_batch)
      : Perceptron(in_dims, 1, max_batch)
    {}

protected:

    /**
     * @short Computes the gradient of the weight vectors and save it in grad_
     *        according to the Oja rule
     * 
     * @param data Data matrix. [batch_size, in_dims]
     * @param batch_size Size of batch preffered to in data.
     *                   batch_size <= max_batch
     * 
     * @return Gradient of the weight vectors according to the specific rule.
     *         The result is saved in grad_
     */
    virtual void backward(const scalar* data, int batch_size) override final;

};
