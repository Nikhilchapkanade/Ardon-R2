# R2 Project — Status Report (Layer-wise Completion)

> Snapshot at v0.1.0. See `ARCHITECTURE.md §10–§11` for the narrative
> summary and `KNOWN_LIMITATIONS.md` for the honest scoping list.

| Layer | % |
|---|---|
| **Frontend** (lexer, parser, AST, REPL) | **95%** |
| **Type Inferencer** (Phase A) | **90%** |
| **R2-IR** (typed SSA, Phase B) | **85%** — closure capture inference conservative |
| **Cranelift JIT** (Phase C) | **75%** — scalar + vector reductions + composed numeric bodies + branchy multi-block IR + 3-arg ternary (ifelse-shape) shipped (Phase C.5); closures/strings → Phase G |
| **Oracle** (Phase E auto-scheduler) | **70%** — V1 dispatch table works; calibration intentionally bundled into Phase G hardware-awareness (cores / ISA / cache); GPU/Cloud placement = V2/V3 |
| **r2-kernel** (reduce/map/binary/ternary/par_for, Phase K) | **100% for v0.1.0 scope** — Phase K.5 (MulAdd `TernaryOp`) + Phase K.6 (`reduce_strided` for non-contiguous matrix slices) shipped this round |
| **r2-linalg** (BLAS L1–L3, decompositions) | **90%** — `dsyev_full` (Householder + Wilkinson-shift QR) shipped; `dgesvd_full` (thin SVD with U + Vᵀ via Bᵀ·B route through `dsyev_full`) shipped; only LAPACK-grade `dbdsqr` (κ-independent SVD) deferred |
| **r2-arrow** (Apache Arrow memory layer) | **85%** — F.3 storage migration (`Reals`/`Ints`/`Logicals` with lazy columnar cache) + F.4 binary kernels + F.5 mmap-backed reader + F.6 dtypes (`ColumnarI32` + packed `ColumnarBool`) all shipped; only `ColumnarI64` / `ColumnarUtf8` / mmap-write deferred |
| **Rayon parallelism** (Phase D) | **95%** — wrapped under kernel; inner tree-split parallelism shipped |
| **r2-stats** (R.0 / R.9 / R.10 / R.11 — reductions, dist, summary, htest, models) | **95%** — full lm/glm/aov/anova migrated to r2-stats::models with glm diagnostics (null deviance, AIC, Fisher iterations, std errors / z values); all 6 hypothesis tests with R-style output, Welch–Satterthwaite df, paired tests, exact hypergeometric Fisher; only Lentz CF for incomplete-beta deferred (current trapezoidal ~1e-4 accuracy is fine for typical workflows) |
| **r2-ml** (R.1 — rpart/rf/gbm/kmeans/knn/naive_bayes/cv/prcomp) | **90%** — all builtins shipped including prcomp via real Householder QR (`dsyev_full`) replacing the covariance hack |
| **r2-data** (R.2 + R.7 — bind/dplyr/apply/table/meta/clean/order) | **90%** — subset/transform NSE shipped (engine pre-processor binds df columns into child env); multi-key merge + Reduce/Filter/Map still pending (v0.2.0) |
| **r2-graphics** (R.3 — plot/hist/boxplot/barplot + overlays) | **60%** — SVG only; PNG/PDF backends + `pairs/image/contour` are v0.2.0 |
| **r2-strings** (R.6) | **90%** — `regex-lite` engine behind default-on `regex` feature; `fixed=TRUE` named arg; only `sprintf` width/precision deferred |
| **r2-io** (R.8) | **85%** — RFC 4180 state-machine parser (embedded separators, doubled quotes, multi-line fields, BOM stripping); `readLines`/`writeLines`/RDS still pending |
| **r2-base** (datasets + linalg_ops, R.4) | **75%** — iris/mtcars/airquality only; more datasets (ToothGrowth/ChickWeight/CO2) v0.2.0 |
| **r2-pkg** (manifest reader + library/require/detach runtime) | **60%** — runtime in engine, package-runtime extraction deferred |
| **r2-engine** (orchestrator residue) | **90%** — line count 7282 → ~4860 (-33%) via R.11/R.12 migrations; only `Error(id/y)` formula NSE deferred (current `id =` named arg covers the same statistical capability) |
| **r2-memory** | **20%** — hooks only; arena allocator not started |
| **GPU dispatcher** | **0%** — Phase G, V2/V3 |
| **FFI Hub** | **0%** — Phase G, V2/V3 |
| **Cloud-RAM shards** | **0%** — V3 |
| **Notebook frontend** | **0%** — Phase 4+ |

---

**v0.1.0 publishable readiness:** ✅ **complete.** 233 tests passing,
clean build, full thin SVD, branchy JIT, RFC 4180 CSV, regex, RNG family
consolidated, models split-handlered, columnar storage migration done,
NA-aware logical ops fixed.

**Overall v1.0 readiness (architecture targets):** ~**75%** (was 60% at
v0.0.9; the remaining 25% is Phase G hardware/GPU/FFI plus v0.2.0+
ergonomics features like graphics backends, multi-key merge, more
datasets).
