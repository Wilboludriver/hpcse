#include <fstream>

namespace utils {

class dataloader {

public:
  dataloader(int size);
  ~dataloader();
  void load();
  void compute_delta();

  int size_;

  double *x;
  double *y;
  double *delta_x;
  double *delta_y;
};

dataloader::dataloader(int size) : size_(size) {
  x = new double[size_]();
  y = new double[size_]();
  delta_x = new double[size_ - 1]();
  delta_y = new double[size_ - 1]();
}

dataloader::~dataloader() {
  delete[] x;
  delete[] y;
  delete[] delta_x;
  delete[] delta_y;
}

void dataloader::load() {

  std::ifstream file;

  // load x
  file.open("data/x_data");
  for (int i = 0; i < size_; i++)
    file >> x[i];
  file.close();

  file.open("data/y_data");
  for (int i = 0; i < size_; i++)
    file >> y[i];
  file.close();
}

void dataloader::compute_delta() {

#pragma omp parallel for
  for (int i = 0; i < size_ - 1; i++) {
    delta_x[i] = x[i + 1] - x[i];
  }

#pragma omp parallel for
  for (int i = 0; i < size_ - 1; i++) {
    delta_y[i] = y[i + 1] - y[i];
  }
}

} // namespace utils