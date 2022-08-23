CXX=g++
CXXFLAGS=-Iinc -I/usr/include/openblas

%.o: src/%.cpp inc/%.hpp
		$(CXX) $(CXXFLAGS) $< -c -o $@

main: main.cpp hebb.o oja.o perceptron.o sanger.o
		$(CXX) $(CXXFLAGS) -o $@ $^ -lopenblas

test: test.cpp perceptron.o sanger.o
		$(CXX) $(CXXFLAGS) -o $@ $^ -lopenblas

unit_test: test
		python3 test_run.py 42

plots: main
		./main
		python3 plot.py

all: main unit_test plots
.DEFAULT_GOAL := main

clean:
	rm *.o *.png main test output/* -rf
