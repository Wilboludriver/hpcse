CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -pedantic -std=c++14 -fopenmp

all: montecarlo

montecarlo: montecarlo.cpp
	$(CXX) $< $(CXXFLAGS) -o montecarlo

run: montecarlo
	./varym $(N)

plot: run
	./plot

clean:
	rm -rf montecarlo out results.png

.PHONY: all clean run plot runplot

