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

<div class="row">
  <div class="col">
    <!-- thrust logo -->
    <img src="/images/logo_thrust.png">
  </div>
  <div class="col">
    <!-- cub logo -->
    <img src="/images/logo_cub.png">
  </div>
  <div class="col">
    <!-- libcu++ logo -->
    <img src="/images/logo_libcupp.png">
  </div>
</div>

<style>
  .row {
    display: flex;
  }
  .col {
    flex: 33.33%;
    padding: 5px;
  }
</style>

<br>

### NVIDIA CCCL in short:

- Three official NVIDIA libraries (Thrust, CUB, `libcu++`)
- A common goal: provide CUDA developers with **tools to write safer and more efficient code**
- Now unified under a single [NVIDIA/cccl repository](https://github.com/NVIDIA/cccl)

---
level: 2
layout: two-cols-header
---

# Thrust and `libcu++`
<br>

::left::

<img src="/images/logo_thrust.png" style="display: block; margin-left: 20px;" width="65%">

<br>

Parallel algorithms library:
- high-level interface
- portability between CPU and GPU
- compatibility with CUDA, TBB, OpenMP

<br>
<br>

::right::

<img src="/images/logo_libcupp.png" style="display: block; margin-left: 50px;" width="58%">

<br>

CUDA C++ Standard Library:
- C++ SL implementation for host and device
- abstractions for CUDA-specific hardware features

<br>
<br>

---
layout: two-cols-header
---

# The CUB Library
<br>

::left::

Collective primitives for each level of the CUDA threading model:

1. <span style="color: #72b300"> **Warp-level primitives** </span>
    - prefix scan, reductions
    - safely specialized for each CUDA architecture
2. <span style="color: #72b300"> **Block-level primitives** </span> 
    - prefix scan, reduction, sort, histogram
    - compatible with arbitrary thread block sizes
3. <span style="color: #72b300"> **Device-level primitives** </span>
    - p 
    - compatible with CUDA dynamic parallelism

<br>

::right::

<img src="/images/logo_cub.png" style="display: block; margin: auto;" width=90%>

---
---

# Warp-wide collectives
<br>

CUB warp-level algorithms are specialized for execution by threads in the same CUDA warp.
These algorithms can only be invoked by `1 <= n <= 32` *consecutive* threads.

- <span style="color: #72b300">`cub::WarpExchange`</span> rearranges data partitioned across a warp
- <span style="color: #72b300">`cub::WarpLoad`</span> loads a linear segment of items from memory into a warp
- <span style="color: #72b300">`cub::` </span>

---
---

# Useful Resources
<br>

- NVIDIA/cccl GitHub repository: https://github.com/NVIDIA/cccl
- CUB official docs page: https://nvidia.github.io/cccl/cub/
- GTC Training on CCCL: https://www.nvidia.com/en-us/on-demand/session/gtcspring21-cwes1801/

---
layout: end
---

# That's all folks!