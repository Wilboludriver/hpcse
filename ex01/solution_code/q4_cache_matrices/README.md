```
To compile and run the ex01 solutions code and plot the results follow the following instructions:
Q4a. Matrix-vector multiplication:

make matrix_vector 

vi Makefile  # to change the -O0 to -O3 and otherwise 

# run the matrix-vector multiplication for matrix size N=100 and for Ns=10 iterations (to compute average timings)
./matrix_vector N Ns
./matrix_vector 100 10 

# note down the timings for the scaling for -O0 and -O3 (to change from no-optimizations to compilation optimzations)

###########################################################

Q4b. Transpose of matrix:

make transpose

./transpose
# Run executable: it will otuput the min, average, max timing for the execution of the transpose operation with:
# - 4 different matrix sizes N = 512, 1024, 2048, 4096
# - non-optimized algorithm + blocking-algorithm for 9 block size choices.
# The executable will also create the transpose_times.txt file, with the minimum recorded times for every combination 
# If you are compiling and running on euler, you can already use the provided bash scripts to run 
# and write your results in the denoted directories:
# the EXTRA flag if you include EXTRA=-DWRITESTRIDE will invoke the code version with writing to strided locations
(./run_transpose <save_dir> EXTRA)
./run_transpose euler
./run_transpose euler_thrash EXTRA=-DWRITESTRIDE

python3 plot_transpose.py 
# plot the performance of transposition vs. the block size vs, the used matrix sizes with python
# Remember that if you execute on euler, you will need to load the python module first.
 
############################################################

Q4c. Matrix-Matric multiplication:

make matrix_matrix

./matrix_matrix  
# The executable will provide you with the average timing for Ns=10 iterations of the matrix-matrix multiplication for:
# 3 matrix sizes: 512, 1024, 2048
# row and column major format 
# non-optimized algorithm and blocking algorithm for 9 block size choices.
# the executable will also create the txt file matrix_matrix_times.txt, which can be used to plot the timings with python:
# to run the executable on euler save in a sub-directory, you can use the bash file:
./run_matrix_matrix 

python3 plot_matrix_matrix.py 
```
