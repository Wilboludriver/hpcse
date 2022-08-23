# Copyright 2021 ETH Zurich. All Rights Reserved.

# TODO a): Set the compiler
MPICXX=
CXXFLAGS+=-std=c++11 -Wall -Wpedantic -O3

run: main
	for i in 2 4 8 16 32 ; do \
	    echo "Running with $$i rank" ; \
	    mpirun -n $$i ./main ; \
	done

main: main.o
	$(MPICXX) $(CXXFLAGS) main.o -o main

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o main
