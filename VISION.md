# R2 Vision — The Path to Green AI

## What R2 Is

R2 is a statistical computing language inspired by R, built from scratch in Rust.
It is NOT an R package, NOT a wrapper, NOT a binding. It is a new language that
speaks R's syntax but runs at Rust's speed.

## The Problem

Modern data science wastes enormous computational resources:

- Python calls C calls Fortran calls CUDA — 4 language boundaries, each with overhead
- R's interpreter executes 50-100x more instructions than necessary
- Installing a typical ML stack: 2-8 GB of dependencies
- Cloud computing burns electricity on interpreter overhead

## R2's Answer

**One language. Minimal glue. Direct to hardware.**

### V0.1.0 — Current Release (2026-05)
- 192 built-in functions; engine 4,860 lines after R.11/R.12 migrations
- 12 ML algorithms built-in — no packages needed
- 2.2x faster matrix multiply than R
- 2.3x faster Random Forest (Rayon parallel)
- 5 MB binary, Rust-only dependencies, no C/C++
- Runs on x86_64 and ARM (Windows, Linux, macOS)
- Full thin SVD with U and Vᵀ; Householder + Wilkinson-shift QR eigendecomp
- JIT compiler with branchy multi-block IR + 3-arg ternary ABI
- RFC 4180 CSV parser; `regex-lite` regex engine; NA-aware `&`/`|`
- Columnar memory layer (F.3–F.6) with mmap-backed reader
- Welch–Satterthwaite df, exact hypergeometric Fisher, glm full diagnostics
- 233 tests passing, clean build
- AGPL v3

### V1.0 — Stability Release
- Bug fixes from community feedback
- More statistical tests and distributions
- Expanded help system and documentation
- Smart auto-parallelism based on data size

### V1.5 — Performance Release
- Rayon parallelism expanded to gbm, cv, kmeans
- Memory-mapped CSV for large file reading
- Expected 4-8x speedup on multi-core workloads

### V2.0 — Bytecode VM (Game-Changer)
- R2 bytecode compiler for user-written functions
- User functions run 10-20x faster than R
- Write in R2 syntax, execute at compiled speed
- .Internal() calls Rust for heavy math — zero overhead
- Community can build statistical packages in R2 — no Rust needed
- This is the release that makes R2 a true language, not just a tool

### V2.5 — Big Data
- Columnar storage engine (inspired by Arrow, built in Rust)
- Process datasets larger than RAM
- Chunked filter/select/aggregate on disk
- Memory-efficient data pipelines

### V3.0 — Universal Compute
- Hardware-accelerated compute for supported platforms
- Distributed computing for cluster deployments
- R2 as a complete data science runtime

## Green AI Impact

| Metric | Traditional Stack | R2 Target |
|---|---|---|
| Install size | 2-8 GB | 5 MB |
| Language boundaries | 3-5 | 1 |
| Interpreter overhead | 50-100x | 0x (compiled) |
| User function speed | baseline (R) | 10-20x faster (bytecode VM) |
| Dependency downloads | hundreds of packages | Rust-only |
| Energy per computation | baseline | 50-70% less |

## Technical Principles

1. **Rust-only dependencies** — no C, C++, or Fortran libraries
2. **No glue code** — one language from script to hardware
3. **Correct first, fast second** — numerical accuracy is non-negotiable
4. **Open source (AGPL v3)** — community-driven development
5. **Green by design** — less overhead = less energy = less carbon

## R2 Roadmap Summary

```
NOW        V0.1.0  →  Ship it. Full SVD, branchy JIT, RFC 4180 CSV, regex,
                       columnar storage. 233 tests passing.
Month 1    V0.2.0  →  Graphics backends (PNG/PDF), multi-key merge,
                       more datasets (ToothGrowth, ChickWeight, CO2),
                       Reduce/Filter/Map, sprintf width/precision.
Month 3    V1.0    →  Stability release. Community feedback baked in.
Month 6    V2.0    →  Phase G — hardware awareness (cores/ISA/cache),
                       Oracle calibration via r2-bench, GPU dispatcher
                       scaffolding (WGPU).
Month 9    V2.5    →  Bytecode VM. User-written functions JITed at the
                       same speed as built-ins.
Month 12   V3.0    →  Universal compute. Distributed processing.
```

## Created By

Devendra Tandale
An AI assisted project
License: AGPL v3
