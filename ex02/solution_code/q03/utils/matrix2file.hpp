#include <fstream>

namespace utils {

/**
 * @brief Writes matrix to file
 *
 * @param Mat Matrix to be stored in a file
 * @param filename Name of the file
 * @param N Number of rows (or leading dimension)
 * @param M Number of columns
 */
void matrix2file(double *Mat, std::string filename, int N, int M);

void matrix2file(double *Mat, std::string filename, int N, int M) {

  std::ofstream file;
  file.open(filename, std::fstream::out);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      file << Mat[j + N * i] << " ";
    }

    file << std::endl;
  }

  file.close();
}

} // namespace utils