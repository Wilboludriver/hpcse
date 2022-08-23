# Overview

In this task we implement a simple reduction kernel.  The kernel works on an
input array and sums up all its elements and returns the scalar sum.

# Workflow


1. Have a look at `gold_red` in `vectorized_reduction.cpp` to get an idea of
   the reduction algorithm.  Continue with implementing the vectorized versions
   `sse_red_4` and `sse_red_2` using the intrinsics from the `xmmintrin.h`
   header.  Note that the two implementations are very similar, implement one
   first and adjust the second.  The [Intel intrinsics
   guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/) is a
   helpful reference for this task.  You only need to look at the SSE
   intinsics.

2. Test your vectorized kernels with the sequential benchmark first to make
   sure that your vectorization works.  This benchmark is already implemented
   for you, just run it (`benchmark_serial`).

3. Once you have everything, run the following `make` target
   ```bash
   make measurement
   ```
   to submit a small test bench on euler that runs for different numbers of
   threads and plots a graph for speedup using two different problem sizes for
   the reduction.  You are then asked to discuss these results in the exercise.
