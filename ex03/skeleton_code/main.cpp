#include <hebb.hpp>
#include <oja.hpp>
#include <perceptron.hpp>
#include <sanger.hpp>
#include <utils.hpp>

#include <fstream>
#include <iostream>
#include <vector>

typedef Perceptron::scalar scalar;

struct JobDescription {
    Perceptron& rule;
    const std::string name;
    const int n_iter;
    const int n_updates_per_iter;
    const scalar lr;
};

int main() {

    // Dataset
    const int N = 1280, D = 1850;
    scalar* data = new scalar[N * D];

    // Load dataset
    std::ifstream infile;
    infile.open("faces.csv");
    load_data(infile, data, N, D);

    // Center dataset
    DataCentering<scalar> centering(D);
    centering.fit(data, N);
    centering.transform(data, N);

    // Set up jobs
    const scalar lr = 1e-2;
    const int n_iter = 150, n_updates_per_iter = 10;

    HebbsRule hebb{D, N};
    OjasRule oja{D, N};
    SangersRule sanger{D, 12, N};

    std::vector<JobDescription> jobs{
        {hebb, "hebb", n_iter, n_updates_per_iter, lr},
        {oja, "oja", n_iter, n_updates_per_iter, lr},
        {sanger, "sanger", n_iter, n_updates_per_iter, lr}
    };

    // Iterate over jobs
    for(JobDescription& job : jobs) {
        
        // Allocate helper structs
        const int k = job.rule.get_out_dims();
        scalar* eigen_values = new scalar[job.n_iter * k];
        scalar* eigen_vectors = new scalar[k * D];

        // Update weights
        for(int i = 0; i < job.n_iter; ++i) {
            for(int j = 0; j < job.n_updates_per_iter; ++j)
                job.rule.update_weights(data, N, job.lr);
            job.rule.get_spectrum(data, N, eigen_values + i * k, nullptr);
        }
        job.rule.get_spectrum(data, N, nullptr, eigen_vectors);
        
        // Save eigen value evolution and eigen vectors
        std::ofstream outfile_values("output/" + job.name + "_eigen_values.csv");
        save_data(outfile_values, eigen_values, job.n_iter, k);
        
        std::ofstream outfile_vectors("output/" + job.name + "_eigen_vectors.csv");
        save_data(outfile_vectors, eigen_vectors, k, D);

        delete[] eigen_values, eigen_vectors;

    }

    delete[] data;

}
