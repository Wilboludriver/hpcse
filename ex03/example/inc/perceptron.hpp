#pragma once
#include <algorithm>

/**
 * @short Perceptron ABC class from which several 
 *        rules are derived to compute 
 *        the eigen values and eigen vectors of a dataset's
 *        empirical covariance matrix
 */
class Perceptron {

public:

    // Typedef defining the type used for computations in this class
    typedef double scalar;

    /**
     * @short Constructor
     *
     * @param in_dims Dimensionality of datapoints in the dataset
     * @param max_batch Upper limit for batch size which can be
     *                  passed to forward, update_weights or get_spectrum  
     */
    Perceptron(int in_dims, int max_batch);

    /**
     * @short Destructor
     */
    ~Perceptron();

    /**
     * @short Computes the matrix product represented by the perceptron
     *        out_ij = data_i * W_j
     *        where W_j is the j-th weight vector of the Perceptron
     *        and data_i is the i-th datapoint
     * 
     * @param data Data matrix. [batch_size, in_dims]
     * @param out Output matrix. [batch_size, in_dims]
     * @param batch_size Size of batch preffered to in data.
     *                   batch_size <= max_batch
     * 
     * @return Matrix product represented by the perceptron
     *         The results are saved in out where out_ij = data_i * W_j
     */
    void forward(const scalar* data, scalar* out, const int batch_size) const;

    /**
     * @short Updates the weights according to the specific gradient rule
     *        Rules deriving from the Perceptron has to specify the 
     *        update rule through the backward method
     * 
     * @param data Data matrix. [batch_size, in_dims]
     * @param batch_size Size of batch preffered to in data.
     *                   batch_size <= max_batch
     * @param lr Learning rate used to update the weights
     */
    void update_weights(const scalar* data, int batch_size, const scalar lr);

    /**
     * @short Computes the intermediate eigen values and eigen vectors
     *        according to the current state of the weights
     * 
     * @param data Data matrix. [batch_size, in_dims]
     * @param batch_size Size of batch preffered to in data.
     *                   batch_size <= max_batch
     * @param eigen_values Output vector for eigen values. [out_dims]
     * @param eigen_vectors Output matrix for eigen vectors of [out_dims, in_dims]
     * 
     * @return Eigen values and eigen vectors according to the current state of the weight vectors.
     *         The results are saved in eigen_values and eigen_vectors respectively
     */
    void get_spectrum(const scalar* data, int batch_size, scalar* eigen_values = nullptr, scalar* eigen_vectors = nullptr) const;

    /**
     * @short Getter for number of input dimensions
     * 
     * @return Number of input dimensions
     */
    int get_in_dims() const;

    /**
     * @short Getter for number of output dimensions
     * 
     * @return Number of output dimensions
     */
    int get_out_dims() const;

    /**
     * @short Getter for upper bound on the batch size
     * 
     * @return Upper bound on the batch size
     */
    int get_max_batch() const;

    /**
     * @short Setter for weights
     * 
     * @param gen Method for generating new weights
     */
    template<typename Func>
    void set_weights(Func&& gen);

    


protected:

    /**
     * @short Normalize the current gradient of each individual weight vector independently
     */
    void normalize_grad();

    /**
     * @short Computes the gradient of the weight vectors and save it in grad_
     *        Has to be specified according to the different rules
     *        Is called by update_weights
     * 
     * @param data Data matrix. [batch_size, in_dims]
     * @param batch_size Size of batch preffered to in data.
     *                   batch_size <= max_batch
     * 
     * @return Gradient of the weight vectors according to the specific rule.
     *         The result is saved in grad_
     */
    virtual void backward(const scalar* data, int batch_size) = 0;

    /**
     * @param in_dims_ Dimensionality of data
     * @param out_dims_ Number of weight vectors. Is fixed to be 1
     * @param max_batch_ Upper limit for batch size which can be
     *                   passed to forward, update_weights or get_spectrum
     */
    const int in_dims_, out_dims_ = 1, max_batch_;

    /**
     * @param weights_ Vector of weights in Perceptron. [in_dims_, out_dims_]
     * @param grad_ Current gradient of weights. [in_dims_, out_dims_]
     * @param outputs_ Helper for storing intermediate results
     */
    scalar *weights_, *grad_, *outputs_;

};


template<typename Func>
void Perceptron::set_weights(Func&& gen) {
    std::generate(weights_, weights_ + in_dims_ * out_dims_ , gen);
}
