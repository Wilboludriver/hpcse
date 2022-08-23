CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -Wpedantic
EXTRA =

targets = matrix_vector transpose  matrix_matrix

all: $(targets)

% : %.cpp
	$(CXX) $< -std=c++11  $(CXXFLAGS) $(EXTRA) -o $@

clean:
	rm -f *.o *~ $(targets) *.txt .tags

.PHONY: all
.PHONY: clean
