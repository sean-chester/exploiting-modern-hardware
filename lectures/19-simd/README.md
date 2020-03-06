# SIMD Parallelisation

## Overview

In this course, you have access to a 6-core multi-core Azure instance. It may surprise you to hear that you already have potential for 8-wide data-level parallelism on _each_ core of your recently-purchased laptop! _SSE/AVX_ is a mature technology that implements SIMD parallelism and is both ubiquitous and broadly unknown. It is a superb example of an "underutili[s]ed installed base" [1] that can provide huge gains at _no_ cost (except development time).

_Single-instruction-multiple-data_ (SIMD) is a parallel paradigm in which we have an array of data on which we want to concurrently apply _the exact same instruction_. In practice, these instructions are taken from a (quickly growing) restricted set of potential instructions, c.f., https://software.intel.com/sites/landingpage/IntrinsicsGuide/#.

This is much more restrictive than, say, multi-core parallelism and parallel reductions, which could tolerate different threads following different execution paths. For example, in multi-core parallelism, there is no harm in two threads taking _divergent_, i.e., opposing, paths at an if statement. SIMD parallelism (like SIMT, i.e., GPU, parallelism to discuss next week) _must_ be branch-free.

The general implementation is to create a really _wide_ register that fits 128 or 256 or 512 bits, rather than the typical 32- or 64-bit registers that house our familiar data types (`float`, `int`, `short`, `double`, `long`). These wide registers can then hold multiple values at once; e.g., a 512-bit-wide register can hold 16 32-bit floats. One can then apply one instruction in that one register which modifies 16 32-bit values all at once. Of course, that requires having 16 32-bit values ready to which we can concurrently apply one instruction.

The idea of packing is not new. We already saw it in the Jacobs paper [2]. What is new is the creation of a custom instruction that treats each lane (i.e., distinct value) independently. In [2], the lanes did not have equal width; age was 7 bits whereas gender was 1 bit. In SIMD, all lanes have equal width, determined by the choice of instruction.

In this (two-part) live-coding lecture, we will work through a basic example of _SIMD'isation_ (a.k.a., _vectorisation_), the process of making code utilise SIMD parallelism. We will focus on the algorithmic and data structure design considerations, rather than a broad coverage of possible SIMD instructions. This lecture design was inspired by [3].


## Problem Definition

Our task for this lesson is to **calculate average vector length**. Let `v = (v.x, v.y, v.z)` be a three-dimensional vector and `|v|` denote the length (i.e., magnitude) of `v`, i.e., `sqrt(v.x^2 + v.y^2 +v.z^2)`. Given a set of vectors `V`, we want to compute the average length:

> Σ_{v∈V}|v|/|V|.

Specifically, we would like to use SIMD to accelerate the computation of this average length.


## Code Instructions

There are two files in this lesson:
  
  * `simd.cpp`: the implementation of a SIMD'ised algorithm for calculating average vector lengths
  * `3d-vectors.hpp`: a static set of 16-byte-aligned, 3d vector structs, generated uniformly at random

To compile the code, you will need the `-mavx` flag to enable SSE/AVX (i.e., SIMD):

> g++ -Wall -O3 -std=c++17 -mavx -march=native simd.cpp -o simd


## Design Sequence

We have gone through two steps to solve this problem:

### (Naive) Single-threaded solution

Beginning from just the main method (benchmarking the solution time), we derive a straight-forward solution that loops over all vectors and adds their length to a running total. This is the solution recorded in the namespace `nosimd::`.

#### Profiling/Analysis

We use `perf` to analyse the performance with "instruction accounting": we record how many instructions are devoted to various types of operations. Among these, we include different types of floating-point instructions:

  * **scalar_single**: Typical, single-value (i.e., non-vectorised) single-precision floating point instruction, like `float p = 2.0f * q;`
  * **scalar_double**: Typical, single-value (i.e., non-vectorised) double-precision floating point instructions, like `double p = 2.0 * q;`
  * **128b_packed_single**: Number of single-precision (i.e., `float` not `double`) instructions performed on 128-bit "packed" registers
  * **256b_packed_single**: Number of single-precision (i.e., `float` not `double`) instructions performed on 256-bit "packed" registers
 
We can find these events under the _floating point_ section of `perf list` events and specify them with the `-e` flag:

> sudo perf stat -e instructions,branch-instructions,mem_inst_retired.all_loads,mem_inst_retired.all_stores,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,f^Carith_inst_retired.256b_packed_single,cycles ./simd

This should produce results similar to:

```
answer: 962.58
time: 13.5092 us

 Performance counter stats for './simd':

     2,621,046,278      instructions              #    3.24  insn per cycle           (44.24%)
       400,386,740      branch-instructions                                           (45.71%)
       598,285,825      mem_inst_retired.all_loads                                     (46.95%)
           826,811      mem_inst_retired.all_stores                                     (46.95%)
     1,387,370,232      fp_arith_inst_retired.scalar_single                                     (46.95%)
            19,989      fp_arith_inst_retired.scalar_double                                     (36.61%)
                 0      fp_arith_inst_retired.128b_packed_single                                     (35.37%)
                 0      fp_arith_inst_retired.256b_packed_single                                     (35.37%)
                 0      fp_arith_inst_retired.128b_packed_double                                     (35.37%)
                 0      fp_arith_inst_retired.256b_packed_double                                     (35.37%)
       810,049,543      cycles                                                        (35.37%)

       0.271687915 seconds time elapsed

       0.267669000 seconds user
       0.003995000 seconds sys
```

In this way, we can confirm that SIMD vectorisation has not been used, as only scalar floating point instructions have been retired. Moreover, we confirm that >50% of the work done by this algorithm are floating point operations and the IPC is very high, 3.24 instructions/cycle, so throughput is high. Together, these two points indicate very strongly that this is a good opportunity to utilise SIMD.

Note that roughly 90% of instructions have been accounted for.

#### Limitations (addressed on Wednesday)

The code that we have produced could benefit from multi-core parallelism (by means of a parallel reduction) and instruction-level parallelism (by means of loop unrolling). However, we do not yet know how to exploit the latent potential 4- or 8-wide data-level parallel potential of the SIMD registers on our machine that are currently idling.


### Naive SIMD'isation

Next we try to _vectorise_ (a.k.a., _simd'ise_) the code. We see that the exactly same operation is applied to the _x_, _y_, and _z_ components of the vector concurrently; i.e., we have latent _single instruction applied to multiple data_ (SIMD) parallelism. We can exploit that by packing an entire 3d vector into one 128-bit register.

We load the data by pointer into a 128-bit data type, via the function: 

> *_mm_load_ps( ptr )*.

This returns a value that corresponds to packing _ptr_ and the 3 subsequent 32-bit floats into the same 128-bit type. It assumes that _ptr_ has been _aligned_ to a 16-byte boundary in advance; otherwise, this line will cause a segmentation fault. Note that we *must* load 128-bits, even if our struct is only 96-bits wide. I.e., this also loads the x-component of the next vector in the array, unavoidably. It is analogous to scalar types: you cannot assign just 28-bits to a 32-bit float. This extra fourth value will be ignored in the end, only serves a purpose of alignment, and is often called "padding."

Next, we apply a vector multiply to square the x, y, and z components all at the same time in one instruction:

> *_mm_mul_ps( vector4, vector4 )*.

This returns another 128-bit type in which each 32-bit lane of the first argument has been multiplied by the same lane of the second argument. After that, the algorithmic requirement is to calculate the square root of the sum of squares: we reinterpret the 128 bit type as an array of 4 floats, and root the sum of the three that we want.

#### Profiling/Analysis

Rerunning `perf`, we observe something similar to:

```
answer: 962.58
time: 14.675 us

 Performance counter stats for './simd':

     2,615,004,169      instructions              #    3.09  insn per cycle           (43.02%)
       403,933,502      branch-instructions                                           (43.39%)
       202,589,624      mem_inst_retired.all_loads                                     (44.75%)
           710,263      mem_inst_retired.all_stores                                     (46.10%)
       795,260,935      fp_arith_inst_retired.scalar_single                                     (47.46%)
            19,916      fp_arith_inst_retired.scalar_double                                     (37.98%)
       199,446,784      fp_arith_inst_retired.128b_packed_single                                     (37.98%)
                 0      fp_arith_inst_retired.256b_packed_single                                     (37.62%)
                 0      fp_arith_inst_retired.128b_packed_double                                     (36.26%)
                 0      fp_arith_inst_retired.256b_packed_double                                     (34.90%)
       845,172,698      cycles                                                        (33.55%)

       0.295206610 seconds time elapsed

       0.295183000 seconds user
       0.000000000 seconds sys
```

Unfortunately, we see a slight degradation of performance. Although we have successfully reduced `1.4e10` scalar floating point operations (FLOPs) to just `8e9` scalar and `2e9` vector FLOPs, a reduction of nearly 30%, and we have reduced overall memory load instructions by a factor of 3, we have not seen a reduction in overall instructions. Evidently, we have introduced some overhead that fully compensated our SIMD gains.

Also note that IPC has *decreased*. This is not surprising, because SIMD can dramatically decrease the number of instructions (i.e., numerator), as we have seen with FLOPs.


#### Limitations (to address on Friday)

Somehow, we failed to observe the gains that we expected when employing SIMD. Despite have 3-wide latent SIMD parallelism, we could not exploit it to effect because of the attendant overhead. On Friday we will attempt a data structure redesign to improve our SIMD speed-up. In particular, we noticed the following limitations of our current approach:

 * We are limited to 3-wide SIMD, even if we have 8-wide or 16-wide lanes on our processor
 * We only SIMD'ised one instruction (the multiplication/squaring); meanwhile, 80% of the FLOPs are still scalar


### SIMD-conscious Redesign

The method above is somewhat dissatisfying, because we do not observe speed-ups and only the multiplications were vectorised. The central challenge is that we _need to expose more SIMD parallelism_. In order to process eight data elements concurrently, we need to restructure our approach to expose parallelism over unique data points (thousands) rather than component dimensions (three).

However, we need _aligned_ data reads to use SIMD effectively; unaligned reads into SIMD registers have high latency which undermines the gains from concurrency. So, this shift in approach to parallelism necessitates a redesign of the data structure. Specifically, we need a _struct-of-arrays_ (SoA) decomposition so that we can load several x- or y- or z-component values at once; i.e., a transpose of the data matrix. This decomposition is statically pre-generated for the lesson in three files: `xvals.hpp`, `yvals.hpp`, and `zvals.hpp`. It is extremely common when trying to expose SIMD parallelism.

This enables the algorithmic redesign in namespace `SoA::simd::`. Each SIMD lane contains the x-, y-, or z-component of a distinct data point. We can perform the addition by adding two registers together at a time, thereby performing the addition on multiple data points at a time (`_mm256_add_ps()` instruction). Also, we can perform the square roots on each sum of squares for each data point (`_mm256_sqrt_ps()` instruction). Finally, because the final results for each data point is stored in a separate lane, we can keep the running total in a vector accumulator as well (the final `_mm256_add_ps()`); we do not scalarise the computation until outside the loop. Nearly every FLOP has been vectorised with full use of all lanes of the SIMD register.

Moreover, the wider exposed SIMD parallelism permits using wider 256-bit registers. We also increase ILP by loading and squaring the x-, y-, and z-values independently of each other. 

#### Profiling/Analysis

Rerunning `perf`, we observe something similar to:

```
answer: 962.58
time: 2.76365 us

 Performance counter stats for './simd':

       284,766,972      instructions              #    1.80  insn per cycle           (35.62%)
        26,753,025      branch-instructions                                           (42.70%)
        78,549,328      mem_inst_retired.all_loads                                     (49.78%)
           528,055      mem_inst_retired.all_stores                                     (56.86%)
           159,755      fp_arith_inst_retired.scalar_single                                     (57.51%)
            19,914      fp_arith_inst_retired.scalar_double                                     (43.14%)
                 0      fp_arith_inst_retired.128b_packed_single                                     (36.05%)
       164,619,622      fp_arith_inst_retired.256b_packed_single                                     (28.97%)
                 0      fp_arith_inst_retired.128b_packed_double                                     (28.33%)
                 0      fp_arith_inst_retired.256b_packed_double                                     (28.33%)
       158,019,334      cycles                                                        (28.33%)

       0.056711689 seconds time elapsed

       0.056772000 seconds user
       0.000000000 seconds sys
```

The first thing to observe is a 14.68 us/2.76 us = 5.3× speed-up _on a single core_, relative to our initial SIMD attempt,
on an already fast program. This indicates a 5.3/8 = 66.5% _efficiency_ in our utilisation of the full 8-wide SIMD width across our entire application. Note that this corresponds almost exactly to the fraction of instructions (1.646e9/2.848e9 = 58%) that are FLOPs. Multi-core and ILP can be independently applied to achieve even greater accelerations.

It is also worth noting that _nearly all_ FLOPs have been effectively converted to AVX 256-bit registers, and SIMD operations now account for nearly 60% of all instructions retired. We take a hit on IPC, because the numerator (# instructions) has decreased about 10-fold, but we see a big gain in total cycles. This emphasises the importance of looking at both ratios and absolute values.

#### Final Notes

We could try to encourage auto-vectorisation by increasing the optimisation level on the `SoA::nosimd::` baseline. The following flag, `-Ofast`, optimises at the expense of floating point accuracy:

> g++ -Wall -Ofast -mavx -march=native -std=c++17 -fopenmp simd.cpp -o simd

```
answer: 962.58
time: 3.28615 us

 Performance counter stats for './simd':

       443,005,670      instructions              #    2.24  insn per cycle           (35.10%)
        25,721,090      branch-instructions                                           (41.08%)
        75,169,632      mem_inst_retired.all_loads                                     (47.05%)
           560,907      mem_inst_retired.all_stores                                     (53.03%)
            39,610      fp_arith_inst_retired.scalar_single                                     (59.00%)
            20,396      fp_arith_inst_retired.scalar_double                                     (46.97%)
                 0      fp_arith_inst_retired.128b_packed_single                                     (41.00%)
       302,592,616      fp_arith_inst_retired.256b_packed_single                                     (35.02%)
                 0      fp_arith_inst_retired.128b_packed_double                                     (29.05%)
                 0      fp_arith_inst_retired.256b_packed_double                                     (23.90%)
       197,447,347      cycles                                                        (23.90%)

       0.067185701 seconds time elapsed

       0.063276000 seconds user
       0.003954000 seconds sys
```

Here, we get a significant conversion to SIMD and a better IPC, but more instructions overall. The compiler _has not_ done as good a job as we have. In the future, this may change as compilers evolve. **But today**, you can walk into an interview and show how you can outperform the gcc compiler!

## References

[1] Sutter (2012) "Welcome to the Jungle: Or, A Heterogeneous Supercomputer in Every Pocket." https://herbsutter.com/welcome-to-the-jungle/ Accessed: 5-Mar-2020.

[2] Jacobs (2009). "The Pathologies of Big Data." Communications of the ACM 52(8). https://dl.acm.org/doi/10.1145/1536616.1536632

[3] Bikker (2017) "Practical SIMD Programming." http://www.cs.uu.nl/docs/vakken/magr/2017-2018/files/SIMD%20Tutorial.pdf Accessed: 5-Mar-2020.
