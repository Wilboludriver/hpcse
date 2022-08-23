#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>

template<typename T>
T* load_data(std::istream & input, T* data, const int N, const int D, const std::string delimiter = ",") {

    // Parse line by line
    std::string line;
    for(int n = 0; n < N; ++n) {

        std::getline(input, line);
        
        // Iterate over line
        for(int d = 0; d < D; ++d) {
            const size_t pos = line.find(delimiter);
            std::stringstream{line.substr(0, pos)} >> data[n * D + d];
            line.erase(0, pos + delimiter.length());
        }
    }

    return data;

}

template<typename T>
void save_data(std::ostream & output, const T* data, const int N, const int D, const std::string delimiter = ",") {
    for(int n = 0; n < N; ++n)
        for(int d = 0; d < D; ++d) {
            
            // Print value
            output << data[n * D + d];

            // Print seperating token
            if(d + 1 < D)
                output << delimiter;
            else
                output << std::endl;

        }

}

template<typename T>
class DataCentering {

public:

    DataCentering(int D)
      : D_(D), mean_(new T[D])
    {}

    ~DataCentering() {
        delete[] mean_;
    }

    void fit(T* data, const int N) {

        // Initialize to zero
        std::fill(mean_, mean_ + D_, 0.0);

        // Compute mean
        for(int n = 0; n < N; ++n)
          for(int d = 0; d < D_; ++d)
              mean_[d] += data[n * D_ + d];
      
        for(int d = 0; d < D_; ++d)
            mean_[d] /= N;

    }

    void transform(T* data, const int N) const {
        
        // Subtract mean from dataset
        for(int n = 0; n < N; ++n)
            for(int d = 0; d < D_; ++d)
                data[n * D_ + d] -= mean_[d];
        
    }

    void inverse_transform(T* data, const int N) const {
        
        // Add mean to dataset
        for(int n = 0; n < N; ++n)
            for(int d = 0; d < D_; ++d)
                data[n * D_ + d] += mean_[d];

    }

    const T* get_mean() const {
        return mean_;
    }

private:
    int D_;
    T* mean_;

};
