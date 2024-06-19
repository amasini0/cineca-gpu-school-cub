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

<Toc minDepth="1" maxDepth="2"></Toc>

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

<div style="width: 100%;">
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

# Why use CUB?
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

# How to use CUB
<br>

<div style="width: 90%">
CUB is part of CCCL, which is <span style="color: #72b300;">included in the CUDA Toolkit</span> (so you already have it). The rest depends on the compiler you want to use:

- When using `nvcc` the relevant header paths are automatically added during compilation, you only need to `#include <cub/cub.h>`

- When using a different compiler, you also need to provide the CUB header path to the compiler using the appropriate flag (e.g. `-I/path/to/cuda/include`)

<br>

If the CCCL version in the toolkit is a bit old you can always <span style="color: #72b300;">get the latest version from GitHub</span> and use it directly like so (see compatibility table [here](https://github.com/NVIDIA/cccl?tab=readme-ov-file#cuda-toolkit-ctk-compatibility))
```bash
git clone https://github.com/NVIDIA/cccl.git
nvcc -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub main.cu -o main
```
</div>

---
---

# Example: block-wide sorting
<br>

---
---

# Warp-wide collectives
<br>

<div style="width: 90%;">
CUB warp-level algorithms are specialized for execution by threads in the same CUDA warp.
These algorithms can only be invoked by `1 <= n <= 32` *consecutive* threads.

- <span style="color: #72b300">`cub::WarpExchange`</span> rearranges data partitioned across a warp
- <span style="color: #72b300">`cub::WarpLoad`</span> loads a linear segment of items from memory into a warp
- <span style="color: #72b300">`cub::WarpMergeSort`</span> sorts items partitioned across a warp
- <span style="color: #72b300">`cub::WarpReduce`</span> computes reduction of items partitioned across a warp
- <span style="color: #72b300">`cub::WarpScan`</span> computes a prefix scan of items partitioned across a warp
- <span style="color: #72b300">`cub::WarpStore`</span> stores items partitioned across a warp to a linear segment of memory
</div>

---
---

# Block-wide collectives
<br>

---
---

# Device-wide collectives
<br>

---
---
# Resources
<br>

<div style="width: 90%;">
Some useful links:

- <span style="color: #72b300;">NVIDIA/cccl GitHub repository</span> 
    - https://github.com/NVIDIA/cccl
- <span style="color: #72b300;">CUB official documentation:</span> 
    - https://nvidia.github.io/cccl/cub/
- <span style="color: #72b300;">GTC Training on CCCL:</span> 
    - https://www.nvidia.com/en-us/on-demand/session/gtcspring21-cwes1801/
</div>

---
layout: end
hideInToc: true
---

# <span style="color: #72b300;">That's all folks! </span>