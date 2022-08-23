config ?= debug

ifeq ($(shell uname -s), Darwin)
CC=gcc-9
else
CC=gcc
endif

LIBPATH= -L$(MKLROOT)/lib/intel64
MKL_LINK= -Wl,--start-group -lmkl_core -lmkl_intel_lp64 -lmkl_gnu_thread -Wl,--end-group -lpthread -lm

INCLUDE= -Iutils/

CPPFLAGS= -std=c++11 -fopenmp
CPPFLAGS+= -fstrict-aliasing -mtune=native  -march=native
CPPFLAGS+= -fmerge-all-constants -fassociative-math

LIBS= -lstdc++ -lm
LIBS+= -lpthread -lz -ldl

ifeq "$(config)" "debug"
CPPFLAGS+= -g -O0
endif

ifeq "$(config)" "prod"
CPPFLAGS += -DNDEBUG -O3 -ffast-math
endif

CPPFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic
CPPFLAGS+= -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CPPFLAGS+= -Woverloaded-virtual -Wno-unused-parameter


main: main.o
	$(CC) $(CPPFLAGS) $(LIBS) $(LIBPATH) $(MKL_LINK) main.o -o $@

all: main
.DEFAULT_GOAL := all

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(INCLUDE) -c $< -o $@


clean:
	rm -rf *.o *.dSYM *.s *.d main b*
