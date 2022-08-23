# File       : Makefile
# Description: Compile targets
# Copyright 2020 ETH Zurich. All Rights Reserved.

CXX ?= g++
CXXFLAGS ?= -Wall -Wextra -pedantic -std=c++11 -O2 -mavx2 -mfma #-DNDEBUG 
ISPC ?= ispc
ISPCFLAGS ?= -O2 -DNDEBUG --opt=disable-assertions --target=avx2-i32x8 --arch=x86-64

.PHONY: clean

gemv: gemv_avx2.o gemv.cpp 
	$(CXX) $(CXXFLAGS) -m64 -o $@ $^

gemv_avx2.o: gemv.ispc
	$(ISPC) $(ISPCFLAGS) $(extra) \
		-o gemv_avx2.o -h gemv_avx2.h $<

clean:
	rm -f gemv gemv_*.h gemv_*.o gemv_*.s

install_ispc_linux_x86_64:
	tar -C ./bin -xJf ./bin/ispc.tar.xz
	mkdir -p ~/.local/bin
	mv ./bin/ispc ~/.local/bin
	echo 'export PATH=$$HOME/.local/bin:$$PATH' >> ~/.bashrc