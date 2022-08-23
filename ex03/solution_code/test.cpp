#include <sanger.hpp>
#include <utils.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>


typedef Perceptron::scalar scalar;

int main(int argc, char* argv[]) {

    if(argc < 6) {
        std::cerr << "Invalid input parameters. Usage: " << std::string(argv[0]) 
                  << " N D k seed out_dir" << std::endl
                  << "N - Number of data points" << std::endl
                  << "D - Dimension of data points" << std::endl
                  << "k - Number of weight vectors" << std::endl
                  << "seed - Seed for RNG" << std::endl
                  << "out_dir - Directory for outputing files" << std::endl;
        exit(1);
    }

    // Get params
    int N, D, k, seed;
    std::stringstream{argv[1]} >> N;
    std::stringstream{argv[2]} >> D;
    std::stringstream{argv[3]} >> k;
    std::stringstream{argv[4]} >> seed;

    std::string out_dir{argv[5]};

    // Set up RNG
    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar> dist {0, 1};
    auto gen = [&dist, &rng](){ return dist(rng); };

    // Set up Sanger rule
    SangersRule sanger{D, k, N};
    sanger.set_weights(gen);

    // Generate data
    scalar* X = new scalar[N * D];
    std::generate(X, X + N * D, gen);

    // ------ Test forward ------

    // Call forward
    scalar* O = new scalar[N * k];
    sanger.forward(X, O, N);

    // Open file
    std::stringstream fname_forward;
    fname_forward << out_dir << "/forward_" << N << "_" << D << "_" << k << "_" << seed << ".csv";
    std::ofstream outfile_forward(fname_forward.str());

    // Save data and clean up
    save_data(outfile_forward, O, N, k);
    delete[] O;

    // ------ Test backward ------

    // Testing the update implicilty by using update_weights
    scalar* W = new scalar[k * D];
    sanger.update_weights(X, N, 1);
    sanger.get_spectrum(X, N, nullptr, W);

    // Open file
    std::stringstream fname_backward;
    fname_backward << out_dir << "/backward_" << N << "_" << D << "_" << k << "_" << seed << ".csv";
    std::ofstream outfile_backward(fname_backward.str());

    // Save data and clean up
    save_data(outfile_backward, W, k, D);
    delete[] W;

}

