config ?= debug
blas ?= mkl

CXX=g++

INCLUDE= -Iutils/

CXXFLAGS= -std=c++11 -fopenmp
CXXFLAGS+= -fstrict-aliasing -mtune=native -march=native
CXXFLAGS+= -fmerge-all-constants

LIBS= -lstdc++ -lm
LIBS+= -lpthread -lz -ldl

ifeq "$(blas)" "mkl"
CXXFLAGS+= -D_MKL_
LIBPATH= -L$(MKLROOT)/lib/intel64
LINK= -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp
endif

ifeq "$(blas)" "openblas"
INCLUDE+= -I/usr/include/openblas 
LINK= -lopenblas
endif

ifeq "$(config)" "debug"
CXXFLAGS+= -g -O0 
endif

ifeq "$(config)" "prod"
CXXFLAGS += -DNDEBUG -O3 -ffast-math
endif

CXXFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic
CXXFLAGS+= -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CXXFLAGS+= -Woverloaded-virtual -Wno-unused-parameter

main: main.o
	$(CXX) $(CXXFLAGS) $(LIBS) $(LIBPATH) $(LINK) main.o -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

run: main
	./main

b%: main
	./main

plot: b_dense b_symmetric b_tridiagonal
	./plot.py

all: plot

clean:
	rm -rf *.o *.dSYM *.s *.d main b*