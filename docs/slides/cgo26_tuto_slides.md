---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  h1 {
    color: #2d5a8b;
  }
  h2 {
    color: #3d7ab5;
  }
  code {
    background-color: #f4f4f4;
  }
---

# XTC: A Research Platform for Optimizing AI Workload Operators

Hugo Pompougnac, Christophe Guillon, Sylvain Noiry,
Alban Dutilleul, Guillaume Iooss, Fabrice Rastello

**Paper:** https://arxiv.org/abs/2512.16512
**GitHub:** https://github.com/xtc-tools/xtc

*INRIA Grenoble (CORSE), France*

---

# The Challenge

## Achieving peak performance on AI operators is hard

- Matrix multiplication, convolution, activations...
- Balance **computation** and **data movement**
- Keep hardware units continuously utilized
- Minimize stalls and idle time

---

# The Automation vs. Manual Tuning Dilemma

| Approach | Pros | Cons |
|----------|------|------|
| **Compiler heuristics** | High productivity | Often fail to reach peak performance |
| **Hand-tuned kernels** | Highest performance | Poor portability, high dev effort |

**Goal:** Expose optimization decisions through *controllable* and *portable* interfaces

---

# Scheduling Languages: The Promise

Allow experts to **script optimization transformations**

- Tiling, fusion, vectorization, parallelization...
- Reduce reliance on opaque compiler heuristics
- Can be driven by humans or autotuners

**Examples:** TVM/TE, Halide, MLIR Transform dialect

---

# The Problem: Fragmentation

Each scheduling language is **locked to its ecosystem**

- TVM Tensor Expressions → TVM only
- Halide → Halide framework only  
- MLIR Transform dialect → MLIR ecosystem only

---

# What's Missing?

> There is currently no unified, user-facing API flexible enough to **decouple scheduling specification from code generation**.

- TVM & MLIR hard to compare on equal footing
- Difficult share scheduling strategies across backends
- No common measurement infrastructure

---

# XTC: Our Proposal

## A research platform that decouples:

1. **Scheduling** -- Common API across compilers
2. **Code generation** -- Multiple backends (TVM, MLIR, ...)
3. **Measurement** -- Cross-platform hardware counters

---

![bg right:50% 70%](architecture.png)

# XTC Architecture

**Entry points (blue):**
- High-level scheduling language
- Unified scheduling API
- Design space exploration

**Backends:**
- TVM, MLIR, extensible...

**Measurement (green):**
- x86, ARM (experimental: Apple Silicon, NVIDIA GPUs)

---

# A Taste of XTC

```python
sch.dims = ['I','J','K']
sch.split(root="mm0", dim="J", segments={"J[0]":0,"J[1]":256})
sch.strip_mine(root="J[0]", dim="K", tiles={"K1": 4})
sch.strip_mine(root="J[0]", dim="J", tiles={"J1": 16})
sch.unroll(root="J[0]", unrolls={"J1": 16, "K1": 4})
sch.vectorize(root="J[0]", axes=["J1"])
```

**Same schedule → TVM or MLIR backend**

---

# Scheduling Primitives

| Primitive | Purpose |
|-----------|---------|
| `strip_mine` | Partition iteration domain into blocks |
| `interchange` | Reorder loops for locality/vectorization |
| `split` | Divide loop into contiguous regions |
| `unroll` | Expose instruction-level parallelism |
| `vectorize` | Map to SIMD resources |
| `parallelize` | Distribute across threads/cores |
| `pack/bufferize` | Improve spatial locality |
| `fuse` | Combine producer/consumer operations |

---

# Higher-Level: Declarative Scheduling

```python
sch.descript({
    'I': [],
    'J[0:256]': {
        'K': [],
        'K#4': ['unroll'],
        'J#16': ['vectorize']
    },
    'J[256:258]': {
        'K': []
    }
})
```

**Describe the target loop nest rather than the transformation sequence**

---

# Why XTC for Research?

1. **Fair comparison** of scheduling strategies across backends
2. **Reproducible measurements** with HW counter access
3. **Identify backend limitations** (e.g., MLIR vectorization issues)
4. **Evaluate performance models** against real hardware
5. **Rapid prototyping** of new scheduling languages

---

# Key Results

- **Matches hand-tuned C** with vector intrinsics
- **High correlation** between TVM and MLIR backends
- **Identified mlir-opt vectorization limitation** on generic convolutions
- **Integration into an existing AI framework** (15-30× speedup over generic C++ in Aidge)

---

# Today's Tutorial

## What you'll learn:

1. Installing XTC (for Linux users)
2. Defining a graph of operators with XTC
3. Compiling an XTC graph and evaluating it
4. Using XTC's scheduling API
5. Defining a schedule declaratively
6. Experimenting with Multiple Schedules
7. Leveraging exploration strategies

---

# Let's Get Started!

**Questions before we dive in?**

Paper: https://arxiv.org/abs/2512.16512
GitHub: https://github.com/xtc-tools/xtc
