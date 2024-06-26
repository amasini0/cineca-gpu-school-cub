---
# General settings
title: NVIDIA Cub - CINECA
theme: default
highlighter: shiki
transition: slide-left
drawings:
  persist: false
mdc: true

# Only for this slide
background: https://www.nvidia.com/content/dam/en-zz/Solutions/studio/products/nvidia-studio-gpu-background-image-spec2-bb770_550-d.jpg
# Apply any unocss classes to the current slide
class: text-left
hideInToc: true
---

# NVIDIA CUB

<!-- H1 color -->
<style>
h1 {
  background-color: #72b300;
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

Reusable software primitives for CUDA programs

<!-- Button to move to next slide -->
<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Start here <carbon:arrow-right class="inline"/>
  </span>
</div>

<!-- Buttons on lower right -->
<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
hideInToc: true
---

# Outline

<br>

<Toc minDepth="1" maxDepth="1"></Toc>

---
---

# C++ Core Compute Libraries
<br>

<div class="ext">
<!-- pictures -->
<div class="row">
  <div class="col">
    <img src="/images/logo_thrust.png">
  </div>
  <div class="col">
    <img src="/images/logo_cub.png">
  </div>
  <div class="col">
    <img src="/images/logo_libcupp.png">
  </div>
</div>
<br>

NVIDIA CCCL in short:

- Three official NVIDIA libraries (Thrust, CUB, `libcu++`)
- A common goal: provide CUDA developers with **tools to write safer and more efficient code**
- Now unified under a single [NVIDIA/cccl repository](https://github.com/NVIDIA/cccl)
</div>

<style>
  .ext { 
    width: 95%;
  }
  .row {
    display: flex;
  }
  .col {
    flex: 33.33%;
    padding: 5px;
  }
</style>
---
level: 2
layout: two-cols-header
---

# Thrust and `libcu++`
<br>

::left::

<div style="width: 95%;">
<img src="/images/logo_thrust.png" style="display: block; margin-left: 20px;" width="65%">

<br>

Parallel algorithms library:
- high-level interface
- portability between CPU and GPU
- compatibility with CUDA, TBB, OpenMP
</div>

::right::

<div style="width: 95%;">
<img src="/images/logo_libcupp.png" style="display: block; margin-left: 50px;" width="58%">

<br>

CUDA C++ Standard Library:
- C++ SL implementation for host and device
- abstractions for CUDA-specific hardware features

</div>

<br>
<br>

---
layout: two-cols-header
---

# The CUB Library
<br>

::left::

<div style="width: 95%;">

Collective primitives for each level of the CUDA threading model:

1. <span style="color: #72b300"> **Warp-level primitives** </span>
    - prefix scan, reduction, sort
    - safely specialized for each CUDA architecture
2. <span style="color: #72b300"> **Block-level primitives** </span> 
    - histogram, adjacent difference, shuffling
    - compatible with arbitrary thread block sizes
3. <span style="color: #72b300"> **Device-level primitives** </span>
    - parallel operations, batched algorithms
    - compatible with CUDA dynamic parallelism

<br>
</div>

::right::

<div style="width: 90%;">

<img src="/images/logo_cub.png" style="display: block; margin: auto;">

</div>


---
level: 2
---

# CUB Features
<br>

<div style="width: 90%;">

There are 4 central features in CUB:

1. <span style="color: #72b300;">**Generic programming**:</span> CUB uses C++ templates to provide flexible, reusable kernel code

2. <span style="color: #72b300;">**Reflective interface**:</span> CUB collectives export their resource requirements (as template parameters or function parameters) to allow compile-time/runtime code tuning

3. <span style="color: #72b300;">**Flexible data arrangement**:</span> CUB collectives operate on data partitioned across groups of threads. Efficiency can be increased by setting partioning granularity or memory arrangement

4. <span style="color: #72b300;">**Static tuning and co-tuning**:</span> most CUB primitives provide alternative algorithmic strategies and variable parallelism levels, which can be coupled to enclosing kernels

</div>

---
level: 2
---

# Why Use CUB?
<br>

<div style="width: 90%;">

A few benefits of using CUB in your kernels:

- <span style="color: #72b300">**Simplicity of composition**:</span> complex parallel operations can be easily sequenced and nested.

- <span style="color: #72b300">**High performance**:</span> CUB algorithms are state-of-the-art

- <span style="color: #72b300">**Adaptable performance**:</span> CUB primitives are specialized to match the diversity of NVIDIA hardware, and can be easily tuned to the available resources

- <span style="color: #72b300">**Increased productivity**:</span> using CUB simplifies writing kernel code, and eases its maintenance

- <span style="color: #72b300">**Code robustness**:</span> CUB Just Works™ with arbitrary data types and widths of parallelism (not limited to C++ types or powers of two threads per block)

</div>

---
level: 2
---

# How to Use CUB
<br>

<div style="width: 90%">

CUB is part of CCCL, which is <span style="color: #72b300;">included in the CUDA Toolkit</span> (so you already have it). The rest depends on the compiler you want to use:

- When using `nvcc` the relevant header paths are automatically added during compilation, you only need to `#include <cub/cub.h>`

- When using a different compiler, you also need to provide the CUB header path to the compiler using the appropriate flag (e.g. `-I/path/to/cuda/include`)

<br>

If the CCCL version in the toolkit is a bit old you can always <span style="color: #72b300;">get the latest version from GitHub</span> and use it directly like so (see compatibility table [here](https://github.com/NVIDIA/cccl?tab=readme-ov-file#cuda-toolkit-ctk-compatibility))

```shell
git clone https://github.com/NVIDIA/cccl.gitsh
nvcc -I cccl/thrust -I cccl/libcudacxx/include -I cccl/cub main.cu -o main.cu
```

</div>

---
---

# Common patterns
<br>

<div style="width: 90%">

CUB’s algorithms are unique at each layer, but offer similar usage experiences:

- they are provided as <span style="color: #72b300">classes</span>,
- they require <span style="color: #72b300">temporary storage</span> for internal data communication,
- they use <span style="color: #72b300">specialized implementations</span> depending on compile-time and runtime information.

<br>

Invoking any CUB algorithm follows the same general pattern:

1. Select the class for the desired algorithm
3. Allocate the temporary storage
4. Pass the temporary storage to the algorithm
5. Invoke it via the appropriate member function


</div>



---
---

# Warp-Wide Collectives
<br>

<div style="width: 90%;"> 

CUB warp-level algorithms are specialized for execution by threads in the same CUDA warp.

- <span style="color: #72b300">`cub::WarpExchange`</span> rearranges data partitioned across a warp
- <span style="color: #72b300">`cub::WarpLoad`</span> loads a linear segment of items from memory into a warp
- <span style="color: #72b300">`cub::WarpMergeSort`</span> sorts items partitioned across a warp
- <span style="color: #72b300">`cub::WarpReduce`</span> computes reduction of items partitioned across a warp
- <span style="color: #72b300">`cub::WarpScan`</span> computes a prefix scan of items partitioned across a warp
- <span style="color: #72b300">`cub::WarpStore`</span> stores items partitioned across a warp to a linear segment of memory

<br>

**Note**: these algorithms can only be invoked by `1 <= n <= 32` *consecutive* threads.

</div>

---
layout: two-cols-header
level: 2
---

# Reductions with <span style="color: #72b300;">`cub::WarpReduce`</span>
<br>

::left::

<div style="max-width: 450px">

```c++
using WarpReducer = cub::WarpReduce<int, threads_per_warp>;

__global__ void warpReduction(int* vec, int* out) {
  // Allocate shared memory for thread communication
  __shared__ WarpReducer::TempStorage temp[warps_per_block];

  if (threadIdx.x % 32 < threads_per_warp) {
    // Assign thread local variables and data
    int warp_lid = threadIdx.x / 32;
    int warp_gid = blockIdx.x * warps_per_block + warp_lid;
    int thread_gid = global_wid * threads_per_warp
                   + threadIdx.x % 32; 
    int thread_data = vec[global_tid];

    // Compute reduction
    int warp_sum = WarpReducer(temp[local_wid])
                  .Sum(thread_data);

    // Output from lane0
    if (threadIdx.x % 32 == 0) out[global_wid] = warp_sum;
  }
}
```

</div>

:: right::

<div style="margin: auto; padding-left: 50px">

**Steps for reduction**:

1. specialize template with data type and logical warp size (max 32),
2. allocate block shared memory for thread communication,
3. initialize `warpReducer` object passing current warp shared memory slot, 
4. reduce to `lane0` using one reduction function.

All available reductions are listed [here](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html).

</div>

<div style="width: 3%; position: fixed; bottom: 30px; right: 65px" align="right"> 
  <a href="https://godbolt.org/z/ebsvWc1vz">
    <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/godbolt_logo_icon_168158.png" width="100%">
  </a>
</div>

---
level: 2
layout: two-cols-header
---

# Prefix Scan with <span style="color: #72b300;">`cub::WarpScan`</span>
<br>

::left::

<div style="max-width: 450px">

```c++
using WarpScanner = cub::WarpScan<int, threads_per_warp>;

__global__ void warpScan(int* vec, int* out, int* agg) {
  // Allocate shared memory for thread communication
  __shared__ WarpScanner::TempStorage temp[warps_per_block];

  if (threadIdx.x % 32 < threads_per_warp) {
    // Assign thread local variables and data
    int warp_lid = threadIdx.x / 32;
    int warp_gid = blockIdx.x * warps_per_block + warp_lid;
    int thread_gid = warp_gid * threads_per_warp 
                   + threadIdx % 32;
    int thread_data = vec[global_tid];
    int thread_prod, warp_aggregate;
    
    // Compute scan inside each warp
    WarpScanner(temp[warp_lid]).InclusiveScan(
      thread_data, thread_prod, op, warp_aggregate);
    // ...
  }
}
```

</div>

::right::

<div style="margin: auto; padding-left: 50px;">

**Steps for scan**:

1. specialize template with data type and logical warp size (max 32),
2. allocate block shared memory for in-warp thread communication,
3. initialize `WarpScanner` object passing current warp shared memory slot,
4. call `.InclusiveScan(...)` or one its variants.

All available variants are listed [here](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpScan.html).

</div>

<div style="width: 3%; position: fixed; bottom: 30px; right: 65px" align="right"> 
  <a href="https://godbolt.org/z/q5bP7bqYG">
    <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/godbolt_logo_icon_168158.png" width="100%">
  </a>
</div>

---
layout: two-cols-header
level:2 
---

# Memory arrangements
<br>

<div style="width: 92%">
Many CUDA kernels have performance benefits if we let each thread process more than one datum. However, these benifits strongly depends on how our data are arranged in memory.

CUB primitives allow us to efficiently manipulate data arranged in two specific formats.
<br>

</div>

::left::

<div style="width: 95%">
<img src="https://nvidia.github.io/cccl/cub/_images/blocked.png" style="padding-left: 50px;">

**Blocked arrangement**: 

- items evenly partitioned in <span style="color: #72b300">consecutive blocks</span>, 
- thread `i` owns the `i`-th block
- <span style="color: #72b300">optimal for algorithms</span>, since each thread can work sequentially

<br>

</div>

::right::

<div style="width: 95%">
<img src="https://nvidia.github.io/cccl/cub/_images/striped.png" style="padding-left: 50px;">

**Striped arrangement**:

- items are partitioned in "stripes" <span style="color: #72b300">separated by a certain stride</span>,
- <span style="color: #72b300">optimal for data movements</span>, since it favors read/write coalescing

<br>

</div>

---
layout: two-cols-header
level: 2
---

# Memory management collectives
<br>

::left::

<div style="max-width: 450px">

```c++
using WarpLoader = cub::WarpLoad<int, items_per_thread, cub::WARP_LOAD_DIRECT, 
                                 threads_per_warp>;
using WarpExchanger = cub::WarpExchange<int, items_per_thread, threads_per_warp>;
using WarpStorerBL = cub::WarpStore<int, items_per_thread, cub::WARP_STORE_DIRECT, 
                                    threads_per_warp>;
using WarpStorerST = cub::WarpStore<int, items_per_thread, cub::WARP_STORE_STRIPED, 
                                    threads_per_warp>;

__global__ void warpExchange(int* vec, int* out1, int* out2) {
  // Allocate shared memory for thread communication
  // ... 
    
  int warp_lid = threadIdx.x / threads_per_warp;
  int warp_gid = blockIdx.x * warps_per_block + warp_lid;
  int warp_offset = warp_gid * threads_per_warp * items_per_thread;
  int thread_data[items_per_thread];

  WarpLoader(ld_temp[warp_lid]).Load(vec + warp_offset, thread_data); 
  WarpExchanger(ex_temp[warp_lid]).BlockedToStriped(thread_data, thread_data);
  WarpStorerBL(bl_temp[warp_lid]).Store(out1 + warp_offset, thread_data);
  WarpStorerST(st_temp[warp_lid]).Store(out2 + warp_offset, thread_data);
}
```

</div>

::right::

<div style="margin: auto; padding-left: 50px">

**Note**:

- Load/store algorithms must be specified in template
- These collectives only allow a power-of-two `threads_per_warp`
- `.Store(...)` and `.Load(...)` need a pointer to first element of warp

</div>

<div style="width: 3%; position: fixed; bottom: 30px; right: 65px" align="right"> 
  <a href="https://godbolt.org/z/MGsT9qvrf">
    <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/godbolt_logo_icon_168158.png" width="100%">
  </a>
</div>

---
level: 2
layout: two-cols-header
---

# Exercise: Warp Sort
<br>

Try to use the warp collective `WarpMergeSort` to sort values inside a warp.

:: left::

<div style="max-width: 450px">

```c++
// TODO
// 1. Specialize templates

__global__ void warpSort(int* vec, int* out) {
  // Custom sort operation
  auto less = [=](const auto& x, const auto& y) { return x < y; };

  // Array for thread-local items
  int thread_data[items_per_thread];  

  // TODO
  // 2. Allocate shared memory for thread communication
  // 3. Set required indices and variables
  // 4. Load thread local data
  // 5. Sort thread local data
  // 6. Write sorted data to output array
}
```

<br>

</div>

::right::

<div style="margin: auto; padding-left: 50px">

**Steps**:

1. Start from the provided template (click on the icon in bottom right corner)
2. Fill the device function in the template. Initially it is like the on the left.
3. If the code is correct, the output values should be sorted.

Check the page for `WarpMergeSort` [here](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpMergeSort.html).

</div>

<div style="width: 3%; position: fixed; bottom: 30px; right: 65px" align="right"> 
  <a href="https://godbolt.org/z/q3TWx3acs">
    <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/godbolt_logo_icon_168158.png" width="100%">
  </a>
</div>

---
level: 2
layout: two-cols-header
---
# Solution: Warp Sort
<br>

::left::

<div style="max-width: 450px">

```c++
using WarpLoader = cub::WarpLoad<int, items_per_thread, cub::WARP_LOAD_VECTORIZE, threads_per_warp>;
using WarpSorter = cub::WarpMergeSort<int, items_per_thread, threads_per_warp>; 
using WarpStorer = cub::WarpStore<int, items_per_thread, cub::WARP_STORE_VECTORIZE, threads_per_warp>;

__global__ void warpSort(int* vec, int* out) {
  auto less = [=](const auto& x, const auto& y) { return x < y; };
  int thread_data[items_per_thread];

  // Allocate shared memory for thread communication
  // ...

  // Assign thread local variables and data
  const int warp_lid = threadIdx.x / threads_per_warp;
  const int warp_gid = blockIdx.x * warps_per_block + warp_lid;
  const int warp_offset = warp_gid * threads_per_warp * items_per_thread;
    
  // Load data, sort them and put them back
  WarpLoader(load_temp[warp_lid]).Load(vec + warp_offset, thread_data);
  WarpSorter(sort_temp[warp_lid]).Sort(thread_data, less);
  WarpStorer(stre_temp[warp_lid]).Store(out + warp_offset, thread_data);
}
```

</div>

::right::

<div style="margin: auto; padding-left: 50px">

**Note**:

- This only works for a power-of-two `threads_per_warp`.
- We use `cub::WARP_LOAD_VECTORIZED` to load blocks of items faster, 
- and analogously for storing sorted blocks.

</div>

<div style="width: 3%; position: fixed; bottom: 30px; right: 65px" align="right"> 
  <a href="https://godbolt.org/z/oajb8r1Mf">
    <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/godbolt_logo_icon_168158.png" width="100%">
  </a>
</div>

---
---

# Block-Wide Collectives
<br>

<div style="width: 92%;">

CUB block-level algorithms are specialized for execution by threads in the same CUDA block.

- Block-level variants for all warp-level algorithms:

    <span style="color: #72b300">`cub::BlockExchange`</span>,
    <span style="color: #72b300">`cub::BlockLoad`</span>, 
    <span style="color: #72b300">`cub::BlockMergeSort`</span>, 
    <span style="color: #72b300">`cub::BlockReduce`</span>, 
    <span style="color: #72b300">`cub::BlockScan`</span>, 
    <span style="color: #72b300">`cub::BlockStore`</span>

- New algorithms:
    - <span style="color: #72b300">`cub::BlockAdjacentDifference`</span> computes the difference between adjacent items 
    - <span style="color: #72b300">`cub::BlockDiscontinuity`</span> flags discontinuities in an ordered set of items
    - <span style="color: #72b300">`cub::BlockHistogram`</span> constructs block-wide histograms from data samples
    - <span style="color: #72b300">`cub::BlockRadixSort`</span> sorts items using radix sorting method
    - <span style="color: #72b300">`cub::BlockRunLengthDecode`</span> decodes a run-length encoded sequence of items
    - <span style="color: #72b300">`cub::Shuffle`</span> shifts or rotates items between threads

</div>

---
layout: two-cols-header
level: 2
---

# Adjacent difference

::left::

::right::

---
layout: two-cols-header
level: 2
---

# Discontinuity

::left::

::right::

---
layout: two-cols-header
level: 2
---

# Histogram

::left::

::right::

---
layout: two-cols-header
level: 2
---

# RunLengthDecode

::left::

::right::

---
layout: two-cols-header
level: 2
---

# Shuffle

::left::

::right::

---
---

# Device-Wide Collectives
<br>

<div style="width: 90%">

CUB device-level single-problem parallel algorithms:

- Device-level variants of some block-level algorithms: 

    <span style="color: #72b300">`cub::DeviceAdjacentDifference`</span>,
    <span style="color: #72b300">`cub::DeviceHistogram`</span>,
    <span style="color: #72b300">`cub::DeviceMergeSort`</span>,
    <span style="color: #72b300">`cub::DeviceRadixSort`</span>,
    <span style="color: #72b300">`cub::DeviceReduce`</span>,
    <span style="color: #72b300">`cub::DeviceScan`</span>

- New algorithms:
    - <span style="color: #72b300">`cub::DeviceFor`</span> provides device-wide, parallel operations for iterating over data residing within device-accessible memory
    - <span style="color: #72b300">`cub::DevicePartition`</span> partitions data residing within device-accessible memory
    - <span style="color: #72b300">`cub::DeviceRunLengthEncode`</span> identifies "runs" of same-valued items within a sequence
    - <span style="color: #72b300">`cub::DeviceSelect`</span> compacts data residing within device-accessible memory

</div>

---
---

# Device-Wide Batched Collectives 
<br>

<div style="width: 92%">

CUB device-level segmented-problem (batched) parallel algorithms:

- <span style="color: #72b300">`cub::DeviceSegmentedSort`</span> computes batched sort across non-overlapping data sequences
- <span style="color: #72b300">`cub::DeviceSegmentedRadixSort`</span> computes batched radix sort across non-overlapping data sequences
- <span style="color: #72b300">`cub::DeviceSegmentedReduce`</span> computes reductions across multiple data sequences
- <span style="color: #72b300">`cub::DeviceCopy`</span> provides device-wide, parallel operations for batched copying of data
- <span style="color: #72b300">`cub::DeviceMemcpy`</span> provides device-wide, parallel operations for batched copying of data

<br>

**Note**: as for non-batched algorithms, data must be within device-accessible memory.
</div>


---
---
# Resources
<br>

<div style="width: 90%;">

Some useful links:

- <span style="color: #72b300;">NVIDIA/cccl GitHub repository</span> 

    - https://github.com/NVIDIA/cccl
- <span style="color: #72b300;">CUB official documentation</span> 

    - https://nvidia.github.io/cccl/cub/
- <span style="color: #72b300;">GTC Training on CCCL</span> 

    - https://www.nvidia.com/en-us/on-demand/session/gtcspring21-cwes1801/

</div>

---
layout: center
hideInToc: true
---

# <span style="color: #72b300;">That's all folks! </span>