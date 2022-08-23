CXX=g++
CXXFLAGS=-Iinc -I/usr/include/openblas

%.o: src/%.cpp inc/%.hpp
		$(CXX) $(CXXFLAGS) $< -c -o $@

main: main.cpp hebb.o oja.o perceptron.o
		$(CXX) $(CXXFLAGS) -o $@ $^ -lopenblas

plots: main
		./main
		python3 plot.py

all: main plots
.DEFAULT_GOAL := main

clean:
	rm *.o *.png main output/* -rf
