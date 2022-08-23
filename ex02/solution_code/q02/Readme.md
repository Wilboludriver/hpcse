# Overview

In this task we implement a matrix-matrix multiplication kernel (GEMM).  We
will study the Intel SPMD program compiler (ISPC) to generate vectorized code
for us (you will appreciate this if you found the previous manual vectorization
of the reduction kernel already tedious).

# Workflow

1. Start with the serial implementation of the GEMM kernel in `gemm.cpp`.  This
   will be your baseline you test your ISPC implementation against
   (`gemm_serial`).  You can compile and test your serial implementation with
   ```bash
   make debug=true gemm_serial
   ```
   you may omit `debug=true` if you do not need the debug symbols in your code.
   Your implementation should return a norm of truth of `255.966` for double
   precision data and `256.211` for single precision data.

2. To start working with ISPC, you need to install it first.  The exercise
   provides a 64bit Linux binary which will work on euler or in your 64bit
   Linux distribution.  To install `ispc` run the command
   ```bash
   make install_ispc_linux_x86_64
   ```
   See the `Makefile` for more information.  After `ispc` has been installed,
   you need to either logout and login again to euler or source your
   `~/.bashrc` file again.  To do this, run 
   ```bash
   source ~/.bashrc
   ```
   Have a look at the output of `ispc --help`.

3. Check the lecture slides and the [getting started
   section](https://ispc.github.io/ispc.html#getting-started-with-ispc) on the
   ISPC documentation page to figure out the function signature of an exported
   ISPC function.  You also need to understand the `varying` and `uniform` rate
   qualifiers when working with ISPC.  Complete the function signature of the
   ISPC kernel in `gemm.ispc`.  Next, complete the `Makefile` to establish the
   foundation to compile your ISPC code.  Have a look at `ispc --help`,
   especially the `--arch` and `--target` options.  We want to target `SSE2`
   and `AVX2` extended instruction sets.  After completion, you should be ready
   to successfully execute the main target (since it is the default, you may
   omit `gemm`)
   ```bash
   make gemm
   ```
   Now you are at the point to start working on the ISPC implementation of the
   GEMM kernel.  Write your ISPC function body in the file `gemm.ispc`.  Use
   your serial implementation as a guide.  This is the core part of this
   exercise.

4. Compile your application code and link to the ISPC generated code with
   ```bash
   make gemm
   ```
   you may want to use `debug=true` while developing your code.  To check the
   benchmark results, you can use the `job` target in the `Makefile`.  It will
   compile your code first if there were changes before the job is submitted.
   For convenience, you may want to work on an interactive node on euler such
   that you do not need to submit jobs all the time (and wait for them to
   complete).  You can check out a single core on a node on euler for
   interactive work (for 1 hour) using
   ```bash
   bsub -n 1 -W 01:00 -Is bash
   ```

   If all went well, your vectorized implementations should be _faster_ than
   your serial baseline.
