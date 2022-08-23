CXX=g++ -O3 -Wall -Wextra -std=c++14 -fopenmp

.PHONY: clean run plot

pse2d: pse2d.cpp
	$(CXX) $< -o $@

clean:
	rm -rf output
	rm -f pse2d movie2d.gif

run: pse2d
	mkdir -p output
	rm -f output/*
	./pse2d 64 0.1 1

plot:
	./plot.py

stat: pse2d
	export OMP_NUM_THREADS=12; ./varym 64 0.005 0

statjob: pse2d
	export OMP_NUM_THREADS=12; bsub -n 12 -W 00:30 ./varym 64 0.005 0

statjobfull: pse2d
	export OMP_NUM_THREADS=12; bsub -n 24 -R fullnode -W 00:30 ./varym 64 0.005 0

plotstat: out
	./plotstat


