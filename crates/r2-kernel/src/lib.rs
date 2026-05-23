//! R2 compute kernels — Phase K.
//!
//! Per docs/ARCHITECTURE.md §4.9 + §5 Phase K:
//!   - Parallelism (Rayon, future GPU/Cloud) lives BELOW this layer.
//!   - Builtins call kernel functions only; they don't see backends.
//!   - Each kernel has serial + Rayon impls today; new backends are
//!     additive (just another `impl ReduceBackend for GpuBackend { ... }`).
//!
//! Phase K spine: reduction kernel only. Element-wise (`map`), binary
//! (`a OP b`), and scan kernels arrive in K.1, K.2, K.3.
//!
//! Locked decisions honoured:
//!   §4.5 Pure-Rust deps only (Rayon qualifies).
//!   §4.7 Backwards-compatible — additive crate, no breaking changes.
//!   §4.9 Rayon stays below this layer.

use r2_oracle::{Backend, Op, Shape};
use rayon::prelude::*;

/// Reduction operations supported by the kernel.
///
/// `Var` / `Sd` use Bessel's correction (sample variance, n-1 divisor) —
/// matches R's `var()` / `sd()`. NA propagates: any null in the input
/// returns `None` (matching R's default `na.rm = FALSE`).
///
/// `Median` requires sorting; serial uses quickselect O(n), Rayon uses
/// `par_sort_by` O(n log n / p).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
    Prod,
    Var,
    Sd,
    Median,
}

/// Backend trait — one impl per execution model. Each backend reduces a
/// buffer of nullable f64s to a scalar. `None` propagates NA.
pub trait ReduceBackend: Send + Sync {
    fn reduce(&self, op: ReduceOp, data: &[Option<f64>]) -> Option<f64>;
}

// ── Serial backend ───────────────────────────────────────────────────

pub struct SerialBackend;

impl ReduceBackend for SerialBackend {
    fn reduce(&self, op: ReduceOp, data: &[Option<f64>]) -> Option<f64> {
        let n = data.len();
        match op {
            ReduceOp::Sum => data.iter().try_fold(0.0_f64, |acc, x| x.map(|v| acc + v)),
            ReduceOp::Mean => {
                if n == 0 { return None; }
                data.iter().try_fold(0.0_f64, |acc, x| x.map(|v| acc + v)).map(|s| s / n as f64)
            }
            ReduceOp::Min => {
                if n == 0 { return None; }
                let mut m = f64::INFINITY;
                for x in data {
                    match x { Some(v) => m = m.min(*v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Max => {
                if n == 0 { return None; }
                let mut m = f64::NEG_INFINITY;
                for x in data {
                    match x { Some(v) => m = m.max(*v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Prod => data.iter().try_fold(1.0_f64, |acc, x| x.map(|v| acc * v)),
            ReduceOp::Var | ReduceOp::Sd => {
                if n < 2 { return None; }
                // Two-pass: mean, then sum of squared deviations.
                let mut sum = 0.0; let mut count = 0usize;
                for x in data {
                    match x { Some(v) => { sum += v; count += 1; } None => return None }
                }
                let mean = sum / count as f64;
                let ss: f64 = data.iter().map(|x| {
                    let v = x.unwrap(); let d = v - mean; d * d
                }).sum();
                let var = ss / (count - 1) as f64;
                Some(if matches!(op, ReduceOp::Sd) { var.sqrt() } else { var })
            }
            ReduceOp::Median => {
                if n == 0 { return None; }
                // Reject if any NA — matches R's default na.rm=FALSE.
                // Use the scratch pool: median is a one-shot
                // materialise-and-discard, perfect fit for buffer
                // recycling.
                let mut buf = r2_memory::scratch_acquire(n);
                let result: Option<f64> = (|| {
                    for x in data {
                        match x { Some(v) => buf.push(*v), None => return None }
                    }
                    let m = buf.len();
                    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
                    if m % 2 == 1 {
                        let (_, mid, _) = buf.select_nth_unstable_by(m / 2, cmp);
                        Some(*mid)
                    } else {
                        let upper_idx = m / 2;
                        let (lower, upper, _) = buf.select_nth_unstable_by(upper_idx, cmp);
                        let upper_val = *upper;
                        let lower_val = lower.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        Some((lower_val + upper_val) / 2.0)
                    }
                })();
                r2_memory::scratch_release(buf);
                result
            }
        }
    }
}

// ── Rayon backend ────────────────────────────────────────────────────
//
// Single-pass NaN-propagation pattern: map None→NaN, reduce, check for NaN
// at the end. Avoids two passes (NA-check + sum) at the cost of producing
// NaN as the in-band null marker during reduction.

pub struct RayonBackend;

impl ReduceBackend for RayonBackend {
    fn reduce(&self, op: ReduceOp, data: &[Option<f64>]) -> Option<f64> {
        let n = data.len();
        match op {
            ReduceOp::Sum => {
                let s: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).sum();
                if s.is_nan() { None } else { Some(s) }
            }
            ReduceOp::Mean => {
                if n == 0 { return None; }
                let s: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).sum();
                if s.is_nan() { None } else { Some(s / n as f64) }
            }
            ReduceOp::Min => {
                if n == 0 { return None; }
                let r = data.par_iter().map(|x| x.unwrap_or(f64::NAN))
                    .reduce(|| f64::INFINITY, f64::min);
                if r.is_nan() { None } else { Some(r) }
            }
            ReduceOp::Max => {
                if n == 0 { return None; }
                let r = data.par_iter().map(|x| x.unwrap_or(f64::NAN))
                    .reduce(|| f64::NEG_INFINITY, f64::max);
                if r.is_nan() { None } else { Some(r) }
            }
            ReduceOp::Prod => {
                let r: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).product();
                if r.is_nan() { None } else { Some(r) }
            }
            ReduceOp::Var | ReduceOp::Sd => {
                if n < 2 { return None; }
                // Single-pass any-NA detection using NaN propagation, then
                // a parallel mean and a parallel sum of squared deviations.
                let s: f64 = data.par_iter().map(|x| x.unwrap_or(f64::NAN)).sum();
                if s.is_nan() { return None; }
                let mean = s / n as f64;
                let ss: f64 = data.par_iter()
                    .map(|x| { let v = x.unwrap(); let d = v - mean; d * d }).sum();
                let var = ss / (n - 1) as f64;
                Some(if matches!(op, ReduceOp::Sd) { var.sqrt() } else { var })
            }
            ReduceOp::Median => {
                if n == 0 { return None; }
                // Strip NAs (defaults match serial path); par_sort the rest.
                let mut buf: Vec<f64> = Vec::with_capacity(n);
                for x in data {
                    match x { Some(v) => buf.push(*v), None => return None }
                }
                let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
                buf.par_sort_by(cmp);
                let m = buf.len();
                Some(if m % 2 == 1 { buf[m/2] } else { (buf[m/2-1] + buf[m/2]) / 2.0 })
            }
        }
    }
}

// ── Top-level dispatcher ─────────────────────────────────────────────

/// Reduce a slice of nullable f64 to a scalar. Backend is chosen by Oracle.
/// This is the public entry point — builtins call this and never see Rayon.
pub fn reduce(op: ReduceOp, data: &[Option<f64>]) -> Option<f64> {
    match r2_oracle::dispatch(Op::Reduction, Shape::n(data.len())) {
        Backend::Serial => SerialBackend.reduce(op, data),
        Backend::Rayon => RayonBackend.reduce(op, data),
    }
}

// ════════════════════════════════════════════════════════════════════
// Element-wise map kernel — Phase K.2
// ════════════════════════════════════════════════════════════════════
//
// `MapOp` covers unary functions that produce one output per input.
// NA preserved: `None` in → `None` out at the same index. Domain errors
// (sqrt of negative, log of non-positive) yield `NaN` per IEEE 754;
// builtins decide whether to surface them as warnings.

/// Element-wise unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOp {
    Sqrt,
    Abs,
    Exp,
    Ln,
    Log2,
    Log10,
    Neg,
    // Phase R.M.1 — trig and transcendental ops. NA-aware, IEEE-754 NaN
    // propagation through every operation. Match CRAN R 4.5 behavior:
    // sin/cos/tan accept radians, asin/acos return NaN outside [-1,1].
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Sign,
    Trunc,
    Expm1,
    Log1p,
}

pub trait MapBackend: Send + Sync {
    fn map(&self, op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>>;
}

#[inline]
fn apply_op(op: MapOp, v: f64) -> f64 {
    match op {
        MapOp::Sqrt  => v.sqrt(),
        MapOp::Abs   => v.abs(),
        MapOp::Exp   => v.exp(),
        MapOp::Ln    => v.ln(),
        MapOp::Log2  => v.log2(),
        MapOp::Log10 => v.log10(),
        MapOp::Neg   => -v,
        MapOp::Sin   => v.sin(),
        MapOp::Cos   => v.cos(),
        MapOp::Tan   => v.tan(),
        MapOp::Asin  => v.asin(),
        MapOp::Acos  => v.acos(),
        MapOp::Atan  => v.atan(),
        MapOp::Sinh  => v.sinh(),
        MapOp::Cosh  => v.cosh(),
        MapOp::Tanh  => v.tanh(),
        MapOp::Sign  => if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else if v == 0.0 { 0.0 } else { f64::NAN },
        MapOp::Trunc => v.trunc(),
        MapOp::Expm1 => v.exp_m1(),
        MapOp::Log1p => v.ln_1p(),
    }
}

impl MapBackend for SerialBackend {
    fn map(&self, op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        data.iter().map(|x| x.map(|v| apply_op(op, v))).collect()
    }
}

impl MapBackend for RayonBackend {
    fn map(&self, op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        data.par_iter().map(|x| x.map(|v| apply_op(op, v))).collect()
    }
}

/// Element-wise map dispatcher. Backend chosen by Oracle (Op::PerElementMap).
pub fn map(op: MapOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::PerElementMap, Shape::n(data.len())) {
        Backend::Serial => SerialBackend.map(op, data),
        Backend::Rayon => RayonBackend.map(op, data),
    }
}

// ════════════════════════════════════════════════════════════════════
// Binary kernel — Phase K.3
// ════════════════════════════════════════════════════════════════════
//
// Element-wise vector⊗vector arithmetic. Both inputs must have matching
// length (R-style recycling is the *caller's* responsibility for now —
// recycling lives at a higher layer because it depends on R-syntax
// semantics, not on numerical kernels). NA in either input → NA out.

/// Element-wise binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
}

#[inline]
fn apply_binop(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Pow => a.powf(b),
        BinaryOp::Mod => a.rem_euclid(b),
    }
}

pub trait BinaryBackend: Send + Sync {
    fn binary(&self, op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>>;
}

impl BinaryBackend for SerialBackend {
    fn binary(&self, op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "binary kernel: length mismatch (caller must recycle)");
        a.iter().zip(b.iter()).map(|(x, y)| match (x, y) {
            (Some(xv), Some(yv)) => Some(apply_binop(op, *xv, *yv)),
            _ => None,
        }).collect()
    }
}

impl BinaryBackend for RayonBackend {
    fn binary(&self, op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "binary kernel: length mismatch (caller must recycle)");
        a.par_iter().zip(b.par_iter()).map(|(x, y)| match (x, y) {
            (Some(xv), Some(yv)) => Some(apply_binop(op, *xv, *yv)),
            _ => None,
        }).collect()
    }
}

/// Element-wise binary dispatcher. Backend chosen by Oracle.
/// Inputs must have equal length — recycling is a higher-layer concern.
pub fn binary(op: BinaryOp, a: &[Option<f64>], b: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::PerElementMap, Shape::n(a.len())) {
        Backend::Serial => SerialBackend.binary(op, a, b),
        Backend::Rayon => RayonBackend.binary(op, a, b),
    }
}

// ════════════════════════════════════════════════════════════════════
// Ternary kernel — Phase K.5 (Tier 4): fused multiply-add and friends.
// ════════════════════════════════════════════════════════════════════
//
// Three-input element-wise ops. Initial members: `MulAdd` (`a*b + c`).
// Useful for BLAS-like inner loops, polynomial evaluation, weighted
// sums, gemm row-update kernels, and as a JIT specialisation target.
// NA propagation: any None among the three inputs at position i → None.

/// Element-wise ternary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryOp {
    /// Fused multiply-add: `a*b + c`. Uses `f64::mul_add` so on hardware
    /// with an FMA instruction the multiply and add are a single rounded
    /// operation (one ulp instead of two). Falls back to scalar `a*b + c`
    /// on platforms without FMA.
    MulAdd,
}

#[inline]
fn apply_ternop(op: TernaryOp, a: f64, b: f64, c: f64) -> f64 {
    match op {
        TernaryOp::MulAdd => a.mul_add(b, c),
    }
}

pub trait TernaryBackend: Send + Sync {
    fn ternary(&self, op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>>;
}

impl TernaryBackend for SerialBackend {
    fn ternary(&self, op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "ternary kernel: a/b length mismatch");
        debug_assert_eq!(a.len(), c.len(), "ternary kernel: a/c length mismatch");
        a.iter().zip(b.iter()).zip(c.iter()).map(|((x, y), z)| match (x, y, z) {
            (Some(xv), Some(yv), Some(zv)) => Some(apply_ternop(op, *xv, *yv, *zv)),
            _ => None,
        }).collect()
    }
}

impl TernaryBackend for RayonBackend {
    fn ternary(&self, op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>> {
        debug_assert_eq!(a.len(), b.len(), "ternary kernel: a/b length mismatch");
        debug_assert_eq!(a.len(), c.len(), "ternary kernel: a/c length mismatch");
        (0..a.len()).into_par_iter().map(|i| match (a[i], b[i], c[i]) {
            (Some(xv), Some(yv), Some(zv)) => Some(apply_ternop(op, xv, yv, zv)),
            _ => None,
        }).collect()
    }
}

/// Element-wise ternary dispatcher. Backend chosen by Oracle.
/// All three inputs must have equal length — caller handles recycling.
pub fn ternary(op: TernaryOp, a: &[Option<f64>], b: &[Option<f64>], c: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::PerElementMap, Shape::n(a.len())) {
        Backend::Serial => SerialBackend.ternary(op, a, b, c),
        Backend::Rayon => RayonBackend.ternary(op, a, b, c),
    }
}

// ════════════════════════════════════════════════════════════════════
// Strided reduction kernel — Phase K.6 (Tier 4).
// ════════════════════════════════════════════════════════════════════
//
// Reduce over a strided view of a slice without copying. Index walk is
//   `data[offset], data[offset+stride], ..., data[offset+(count-1)*stride]`.
//
// Motivation: column-major matrices store columns contiguously (stride 1)
// but rows non-contiguously (stride = nrow). Reducing a row required
// a copy-into-`Vec` round-trip; this kernel skips that allocation.
//
// Implementation notes:
//   - Same NA propagation as `reduce`: any `None` in the walked positions
//     → `None`.
//   - Variance/Sd use the same two-pass algorithm but iterate strided
//     indices, not over a copied buffer.
//   - Median still materialises a `Vec<f64>` because select_nth_unstable
//     needs contiguous storage; the win for Median is amortised by
//     avoiding the option-unwrap pass on a temporary.

pub trait StridedReduceBackend: Send + Sync {
    fn reduce_strided(
        &self,
        op: ReduceOp,
        data: &[Option<f64>],
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Option<f64>;
}

impl StridedReduceBackend for SerialBackend {
    fn reduce_strided(
        &self,
        op: ReduceOp,
        data: &[Option<f64>],
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Option<f64> {
        debug_assert!(stride > 0, "stride must be > 0");
        if count == 0 { return None; }
        // Index iterator for the walk.
        let idx = |k: usize| offset + k * stride;
        // Bounds-check the last index up front.
        if idx(count - 1) >= data.len() { return None; }

        match op {
            ReduceOp::Sum => {
                let mut acc = 0.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => acc += v, None => return None }
                }
                Some(acc)
            }
            ReduceOp::Mean => {
                let mut acc = 0.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => acc += v, None => return None }
                }
                Some(acc / count as f64)
            }
            ReduceOp::Min => {
                let mut m = f64::INFINITY;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => m = m.min(v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Max => {
                let mut m = f64::NEG_INFINITY;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => m = m.max(v), None => return None }
                }
                Some(m)
            }
            ReduceOp::Prod => {
                let mut acc = 1.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => acc *= v, None => return None }
                }
                Some(acc)
            }
            ReduceOp::Var | ReduceOp::Sd => {
                if count < 2 { return None; }
                let mut sum = 0.0_f64;
                for k in 0..count {
                    match data[idx(k)] { Some(v) => sum += v, None => return None }
                }
                let mean = sum / count as f64;
                let mut ss = 0.0_f64;
                for k in 0..count {
                    let v = data[idx(k)].unwrap();
                    let d = v - mean; ss += d * d;
                }
                let var = ss / (count - 1) as f64;
                Some(if matches!(op, ReduceOp::Sd) { var.sqrt() } else { var })
            }
            ReduceOp::Median => {
                let mut buf = r2_memory::scratch_acquire(count);
                let result: Option<f64> = (|| {
                    for k in 0..count {
                        match data[idx(k)] { Some(v) => buf.push(v), None => return None }
                    }
                    let m = buf.len();
                    let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap();
                    if m % 2 == 1 {
                        let (_, mid, _) = buf.select_nth_unstable_by(m / 2, cmp);
                        Some(*mid)
                    } else {
                        let (_, hi, _) = buf.select_nth_unstable_by(m / 2, cmp);
                        let hi_v = *hi;
                        let lo_v = buf[..m / 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        Some((lo_v + hi_v) / 2.0)
                    }
                })();
                r2_memory::scratch_release(buf);
                result
            }
        }
    }
}

impl StridedReduceBackend for RayonBackend {
    fn reduce_strided(
        &self,
        op: ReduceOp,
        data: &[Option<f64>],
        offset: usize,
        stride: usize,
        count: usize,
    ) -> Option<f64> {
        debug_assert!(stride > 0, "stride must be > 0");
        if count == 0 { return None; }
        if offset + (count - 1).saturating_mul(stride) >= data.len() { return None; }
        // Two-pass: scan for any NA in parallel; if clean, parallel reduce.
        // (Rayon's try_reduce works on `Try`-implementing items, so we
        // wrap as `Result<f64, ()>` and unwrap via `.ok()` at the end.)
        let any_na = (0..count).into_par_iter()
            .any(|k| data[offset + k * stride].is_none());
        if any_na { return None; }
        match op {
            ReduceOp::Sum => {
                Some((0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .sum::<f64>())
            }
            ReduceOp::Mean => {
                let s: f64 = (0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .sum();
                Some(s / count as f64)
            }
            ReduceOp::Min => {
                (0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .reduce(|| f64::INFINITY, f64::min)
                    .into()
            }
            ReduceOp::Max => {
                (0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .reduce(|| f64::NEG_INFINITY, f64::max)
                    .into()
            }
            ReduceOp::Prod => {
                Some((0..count).into_par_iter()
                    .map(|k| data[offset + k * stride].unwrap())
                    .product::<f64>())
            }
            _ => SerialBackend.reduce_strided(op, data, offset, stride, count),
        }
    }
}

/// Strided reduction dispatcher. Backend chosen by Oracle on the walked
/// element count (`count`), not the underlying slice length.
pub fn reduce_strided(
    op: ReduceOp,
    data: &[Option<f64>],
    offset: usize,
    stride: usize,
    count: usize,
) -> Option<f64> {
    match r2_oracle::dispatch(Op::Reduction, Shape::n(count)) {
        Backend::Serial => SerialBackend.reduce_strided(op, data, offset, stride, count),
        Backend::Rayon => RayonBackend.reduce_strided(op, data, offset, stride, count),
    }
}

// ════════════════════════════════════════════════════════════════════
// Parallel-for kernel — Phase K.4
// ════════════════════════════════════════════════════════════════════
//
// Backend-dispatched parallel-for-each. Caller passes the work `kind`
// (used by Oracle to pick the threshold) and a closure indexed by `i`;
// kernel runs the closure for each `i` in `0..n` and collects results.
// Builtins that previously called `(0..n).into_par_iter().map(...)`
// directly now call `par_for(kind, n, f)` — Rayon stays below this
// layer (§4.9 locked decision).
//
// SAFETY: closures must be `Send + Sync`. Output type must be `Send`.
// Order of result indexing is preserved (Rayon's collect is stable).

/// Parallel-for-each. Backend chosen by Oracle. Result is a `Vec<T>`
/// indexed by `0..n`.
pub fn par_for<T, F>(kind: Op, n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Send + Sync,
    T: Send,
{
    match r2_oracle::dispatch(kind, Shape::n(n)) {
        Backend::Serial => (0..n).map(f).collect(),
        Backend::Rayon => (0..n).into_par_iter().map(f).collect(),
    }
}

/// Force parallel iteration without consulting Oracle. Used by callers
/// that have already made the dispatch decision themselves (e.g. the
/// list-aware apply path that computes aggregate work across
/// heterogeneous components and decides via `Op::ListMap` separately).
pub fn par_for_rayon<T, F>(n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Send + Sync,
    T: Send,
{
    (0..n).into_par_iter().map(f).collect()
}

// ════════════════════════════════════════════════════════════════════
// Phase K.7 — Scan / cumulative operations
// ════════════════════════════════════════════════════════════════════
//
// Element-wise prefix-style reductions: each output position holds the
// reduction of all input positions up to and including it. Common in
// stats (running totals, density integration, cumulative regression).
//
//   cumsum:  out[i] = sum(in[0..=i])
//   cumprod: out[i] = prod(in[0..=i])
//   cummax:  out[i] = max(in[0..=i])
//   cummin:  out[i] = min(in[0..=i])
//
// NA propagation: once a None is seen, every subsequent output is None
// (matches R's `cumsum(c(1, NA, 3))` → `c(1, NA, NA)`).
//
// Serial: trivial O(n) loop.
// Rayon: two-pass Blelloch-style parallel scan. For workloads ≥ ~10K
//   the parallel pass amortises the chunk-merge overhead.

/// Cumulative / scan operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOp {
    /// `out[i] = sum(in[0..=i])`
    Cumsum,
    /// `out[i] = prod(in[0..=i])`
    Cumprod,
    /// `out[i] = max(in[0..=i])`
    Cummax,
    /// `out[i] = min(in[0..=i])`
    Cummin,
}

#[inline]
fn scan_identity(op: ScanOp) -> f64 {
    match op {
        ScanOp::Cumsum => 0.0,
        ScanOp::Cumprod => 1.0,
        ScanOp::Cummax => f64::NEG_INFINITY,
        ScanOp::Cummin => f64::INFINITY,
    }
}

#[inline]
fn scan_combine(op: ScanOp, acc: f64, v: f64) -> f64 {
    match op {
        ScanOp::Cumsum => acc + v,
        ScanOp::Cumprod => acc * v,
        ScanOp::Cummax => acc.max(v),
        ScanOp::Cummin => acc.min(v),
    }
}

pub trait ScanBackend: Send + Sync {
    fn scan(&self, op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>>;
}

impl ScanBackend for SerialBackend {
    fn scan(&self, op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        let mut out = Vec::with_capacity(data.len());
        let mut acc = scan_identity(op);
        let mut hit_na = false;
        for x in data {
            if hit_na { out.push(None); continue; }
            match x {
                Some(v) => {
                    acc = scan_combine(op, acc, *v);
                    out.push(Some(acc));
                }
                None => { hit_na = true; out.push(None); }
            }
        }
        out
    }
}

impl ScanBackend for RayonBackend {
    fn scan(&self, op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
        // Two-pass parallel scan:
        //   Pass 1: split into chunks, reduce each chunk in parallel.
        //   Sequential merge: prefix-combine chunk totals.
        //   Pass 2: re-scan each chunk in parallel, seeded with its prefix.
        //
        // NA handling: a None in chunk k poisons everything from that
        // position onwards. We track per-chunk "first NA index" so the
        // pass-2 scan emits None for the rest of that chunk and all
        // subsequent chunks.
        let n = data.len();
        if n == 0 { return Vec::new(); }
        // For small inputs the parallel overhead loses — fall through to serial.
        if n < 4096 { return SerialBackend.scan(op, data); }

        let n_chunks = num_chunks(n);
        let chunk_size = n.div_ceil(n_chunks);

        // Pass 1: per-chunk reduction + first-NA index.
        #[derive(Clone, Copy)]
        struct ChunkInfo {
            total: f64,
            first_na: Option<usize>, // relative to chunk start
        }
        let infos: Vec<ChunkInfo> = (0..n_chunks).into_par_iter().map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let mut acc = scan_identity(op);
            let mut first_na = None;
            for (i, v) in data[start..end].iter().enumerate() {
                match v {
                    Some(x) => { acc = scan_combine(op, acc, *x); }
                    None => { first_na = Some(i); break; }
                }
            }
            ChunkInfo { total: acc, first_na }
        }).collect();

        // Sequential prefix combine over chunk totals — gives the
        // "scan up to but not including chunk c" seed.
        let mut prefixes = Vec::with_capacity(n_chunks);
        let mut acc = scan_identity(op);
        let mut prefix_na_at = None;
        for (c, info) in infos.iter().enumerate() {
            prefixes.push(acc);
            if prefix_na_at.is_none() && info.first_na.is_some() {
                prefix_na_at = Some(c);
            }
            if prefix_na_at.is_some() {
                // Once any chunk has NA, all subsequent prefixes are
                // "NA from offset 0" — we represent this with a poison.
                acc = scan_identity(op); // doesn't matter; will be masked
            } else {
                acc = scan_combine(op, acc, info.total);
            }
        }

        // Pass 2: per-chunk re-scan seeded with the chunk's prefix.
        let chunks: Vec<Vec<Option<f64>>> = (0..n_chunks).into_par_iter().map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let seed = prefixes[c];
            // If any prior chunk had NA, the whole of this chunk is None.
            let chunk_poisoned = prefix_na_at.map_or(false, |na_c| c > na_c);
            let mut out = Vec::with_capacity(end - start);
            let mut acc = seed;
            let mut hit_na = chunk_poisoned;
            for (i, v) in data[start..end].iter().enumerate() {
                if hit_na { out.push(None); continue; }
                match v {
                    Some(x) => {
                        acc = scan_combine(op, acc, *x);
                        out.push(Some(acc));
                    }
                    None => { hit_na = true; out.push(None); let _ = i; }
                }
            }
            out
        }).collect();

        chunks.into_iter().flatten().collect()
    }
}

/// Compute a reasonable chunk count: roughly one chunk per core, but
/// bounded so chunks aren't pointless-tiny.
#[inline]
fn num_chunks(n: usize) -> usize {
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
    let min_chunk = 1024;
    let max_chunks = (n + min_chunk - 1) / min_chunk;
    cores.min(max_chunks).max(1)
}

/// Public scan dispatcher. Oracle decides Serial vs Rayon based on
/// input length (`Op::Reduction` threshold reused — scan has similar
/// memory-bandwidth profile to reduction).
pub fn scan(op: ScanOp, data: &[Option<f64>]) -> Vec<Option<f64>> {
    match r2_oracle::dispatch(Op::Reduction, Shape::n(data.len())) {
        Backend::Serial => SerialBackend.scan(op, data),
        Backend::Rayon => RayonBackend.scan(op, data),
    }
}

// ════════════════════════════════════════════════════════════════════
// Phase K.8 — Select / find operations
// ════════════════════════════════════════════════════════════════════
//
// Reductions that return positions (indices) or partial orderings
// rather than aggregates:
//
//   which_max / which_min: 0-based index of the first max/min
//   nth_smallest(k):       value of the kth smallest (quickselect)
//   top_k(k) / bottom_k:   indices of the k largest / smallest
//
// NA handling: any None in the input propagates the appropriate
// "no answer" (returns None for index-returning ops; skips None for
// quickselect-style ops to match R's na.rm=TRUE default on these).

/// Index of the first maximum. `None` if input has any NA or is empty.
/// (R's `which.max(c(1, NA, 3))` returns `integer(0)`; we return None
/// to match the broader NA-propagation pattern of the kernel layer.)
pub fn which_max(data: &[Option<f64>]) -> Option<usize> {
    if data.is_empty() { return None; }
    let mut best_idx = 0usize;
    let mut best_val = match data[0] { Some(v) => v, None => return None };
    for (i, v) in data.iter().enumerate().skip(1) {
        match v {
            Some(x) => { if *x > best_val { best_val = *x; best_idx = i; } }
            None => return None,
        }
    }
    Some(best_idx)
}

/// Index of the first minimum. Same semantics as `which_max`.
pub fn which_min(data: &[Option<f64>]) -> Option<usize> {
    if data.is_empty() { return None; }
    let mut best_idx = 0usize;
    let mut best_val = match data[0] { Some(v) => v, None => return None };
    for (i, v) in data.iter().enumerate().skip(1) {
        match v {
            Some(x) => { if *x < best_val { best_val = *x; best_idx = i; } }
            None => return None,
        }
    }
    Some(best_idx)
}

/// Quickselect: returns the value of the `k`-th smallest element (0-indexed).
/// Skips NAs. Returns `None` if all NA or `k >= count of non-NA`.
/// O(n) average, O(n²) worst case but in practice fine since stdlib's
/// `select_nth_unstable_by` uses a hardened pivot strategy.
///
/// Uses the scratch pool for the unwrapped buffer.
pub fn nth_smallest(data: &[Option<f64>], k: usize) -> Option<f64> {
    let mut buf = r2_memory::scratch_acquire(data.len());
    for x in data { if let Some(v) = x { buf.push(*v); } }
    let result = if k >= buf.len() {
        None
    } else {
        let (_, mid, _) = buf.select_nth_unstable_by(k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(*mid)
    };
    r2_memory::scratch_release(buf);
    result
}

/// Indices of the `k` largest elements, in descending order of value.
/// Skips NAs. Uses a binary heap of size `k` — O(n log k) regardless of k.
/// Returns at most `min(k, len-na_count)` indices.
pub fn top_k(data: &[Option<f64>], k: usize) -> Vec<usize> {
    if k == 0 { return Vec::new(); }
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    // Min-heap of (value, idx) — keeps the k largest.
    // Wrap f64 in a NaN-safe ord helper.
    #[derive(PartialEq)]
    struct OrdF64(f64, usize);
    impl Eq for OrdF64 {}
    impl Ord for OrdF64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for OrdF64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
    }
    let mut heap: BinaryHeap<Reverse<OrdF64>> = BinaryHeap::with_capacity(k + 1);
    for (i, v) in data.iter().enumerate() {
        if let Some(x) = v {
            if heap.len() < k {
                heap.push(Reverse(OrdF64(*x, i)));
            } else if let Some(Reverse(OrdF64(top, _))) = heap.peek() {
                if x > top {
                    heap.pop();
                    heap.push(Reverse(OrdF64(*x, i)));
                }
            }
        }
    }
    // Extract in descending order.
    let mut result: Vec<(f64, usize)> = heap.into_iter()
        .map(|Reverse(OrdF64(v, i))| (v, i))
        .collect();
    result.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    result.into_iter().map(|(_, i)| i).collect()
}

/// Indices of the `k` smallest elements, in ascending order of value.
/// Mirror of `top_k`.
pub fn bottom_k(data: &[Option<f64>], k: usize) -> Vec<usize> {
    if k == 0 { return Vec::new(); }
    use std::collections::BinaryHeap;
    #[derive(PartialEq)]
    struct OrdF64(f64, usize);
    impl Eq for OrdF64 {}
    impl Ord for OrdF64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for OrdF64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
    }
    // Max-heap — keeps the k smallest.
    let mut heap: BinaryHeap<OrdF64> = BinaryHeap::with_capacity(k + 1);
    for (i, v) in data.iter().enumerate() {
        if let Some(x) = v {
            if heap.len() < k {
                heap.push(OrdF64(*x, i));
            } else if let Some(OrdF64(top, _)) = heap.peek() {
                if x < top {
                    heap.pop();
                    heap.push(OrdF64(*x, i));
                }
            }
        }
    }
    let mut result: Vec<(f64, usize)> = heap.into_iter()
        .map(|OrdF64(v, i)| (v, i))
        .collect();
    result.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    result.into_iter().map(|(_, i)| i).collect()
}

// ════════════════════════════════════════════════════════════════════
// Phase K.9 — Rolling window operations
// ════════════════════════════════════════════════════════════════════
//
// Fixed-width window reductions: for window size `w`, output position
// `i` (with i in `w-1..n`) holds the reduction over `data[i-w+1..=i]`.
// Output is shorter than input by `w-1` (matches R's `zoo::rollapply`
// with align="right", no padding).
//
//   rollsum:  sliding sum
//   rollmean: sliding mean
//   rollmax:  sliding maximum (deque-based, O(n))
//   rollmin:  sliding minimum (deque-based, O(n))
//   rollsd:   sliding sample standard deviation (two-pass per window)
//
// Common in time-series stats: moving averages, rolling volatility,
// running extremes.
//
// NA semantics: if any element in the window is None, that output is None.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingOp {
    Sum,
    Mean,
    Max,
    Min,
    Sd,
}

/// Rolling-window reduction. Output length = `data.len() - window + 1`
/// if `window <= data.len()`, otherwise empty. Window of 0 returns empty.
pub fn rolling(op: RollingOp, data: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    if window == 0 || data.len() < window { return Vec::new(); }
    let out_len = data.len() - window + 1;
    let mut out = Vec::with_capacity(out_len);

    match op {
        RollingOp::Sum | RollingOp::Mean => {
            // Sliding sum with incremental update. NA tracking: keep
            // `na_count` for the current window; when ≥1, emit None.
            let mut sum = 0.0_f64;
            let mut na_count: usize = 0;
            // Initial window [0..window].
            for v in &data[..window] {
                match v { Some(x) => sum += x, None => na_count += 1 }
            }
            let denom = window as f64;
            let push = |out: &mut Vec<Option<f64>>, sum: f64, na: usize| {
                if na > 0 { out.push(None); }
                else if matches!(op, RollingOp::Mean) { out.push(Some(sum / denom)); }
                else { out.push(Some(sum)); }
            };
            push(&mut out, sum, na_count);
            for i in window..data.len() {
                // Drop leftmost, add rightmost.
                match data[i - window] {
                    Some(x) => sum -= x,
                    None => na_count -= 1,
                }
                match data[i] {
                    Some(x) => sum += x,
                    None => na_count += 1,
                }
                push(&mut out, sum, na_count);
            }
        }
        RollingOp::Max | RollingOp::Min => {
            // Deque-based O(n) sliding extremum. Each element enters
            // and leaves the deque at most once.
            // Index-only deque; values fetched from data[].
            // NA: when window contains any None, emit None and reset.
            use std::collections::VecDeque;
            let mut deque: VecDeque<usize> = VecDeque::new();
            let cmp = |a: f64, b: f64| -> bool {
                if matches!(op, RollingOp::Max) { a >= b } else { a <= b }
            };
            // Walk index, building deque of "candidates": each new index
            // pops back items with worse-or-equal value (they can never
            // be the answer once a better/equal-newer one is in).
            // NA-handling: track most-recent NA index.
            let mut last_na: Option<usize> = None;
            for i in 0..data.len() {
                match data[i] {
                    None => { last_na = Some(i); deque.clear(); }
                    Some(x) => {
                        while let Some(&back) = deque.back() {
                            if cmp(x, data[back].unwrap()) { deque.pop_back(); }
                            else { break; }
                        }
                        deque.push_back(i);
                    }
                }
                // Once we have i >= window-1, emit the answer.
                if i + 1 >= window {
                    let win_start = i + 1 - window;
                    // Drop fronts that are no longer in window.
                    while let Some(&front) = deque.front() {
                        if front < win_start { deque.pop_front(); }
                        else { break; }
                    }
                    let window_has_na = last_na.map_or(false, |na_i| na_i >= win_start);
                    if window_has_na || deque.is_empty() {
                        out.push(None);
                    } else {
                        out.push(Some(data[*deque.front().unwrap()].unwrap()));
                    }
                }
            }
        }
        RollingOp::Sd => {
            // Two-pass sample SD per window. Simpler than incremental
            // (Welford-style) variance and avoids numerical drift on
            // long windows.
            for i in 0..out_len {
                let win = &data[i..i + window];
                let mut na_count = 0usize;
                let mut sum = 0.0_f64;
                for v in win {
                    match v { Some(x) => sum += x, None => { na_count += 1; } }
                }
                if na_count > 0 || window < 2 { out.push(None); continue; }
                let mean = sum / window as f64;
                let mut ss = 0.0_f64;
                for v in win {
                    let x = v.unwrap();
                    let d = x - mean;
                    ss += d * d;
                }
                out.push(Some((ss / (window - 1) as f64).sqrt()));
            }
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════
// Phase K.10 — Hash aggregation
// ════════════════════════════════════════════════════════════════════
//
// Group-by primitive: given parallel `keys` and `values` slices, group
// values by their key and reduce each group. Generic over an `AggOp`
// so `table()`, `tapply()`, group-mean, group-sum, etc. all share one
// kernel.
//
// Implementation: stdlib `HashMap<u64, Accumulator>`. The hashing cost
// is O(n); typical group counts are O(√n) so total memory is sub-linear.
//
// Why a kernel: previously `table()` and similar built their own
// hashing in builtins, missing the parallel dispatch path and
// duplicating bookkeeping logic.

/// Group-by reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    /// Sum of values per group.
    Sum,
    /// Arithmetic mean of values per group.
    Mean,
    /// Count of (non-NA) values per group.
    Count,
    /// Minimum value per group.
    Min,
    /// Maximum value per group.
    Max,
}

/// Result of a hash-aggregation: parallel arrays of unique keys and
/// their reduced values. Order is insertion order (first occurrence of
/// each key in the input).
#[derive(Debug, Clone)]
pub struct HashAggResult {
    pub keys: Vec<u64>,
    pub values: Vec<Option<f64>>,
}

/// Aggregate `values` grouped by `keys` (both same length, parallel
/// arrays). NA values are skipped (na.rm=TRUE behavior); NA keys cause
/// their values to be skipped too.
pub fn hash_agg(
    op: AggOp,
    keys: &[Option<u64>],
    values: &[Option<f64>],
) -> HashAggResult {
    assert_eq!(keys.len(), values.len(), "hash_agg: keys/values length mismatch");
    use std::collections::HashMap;

    // Per-group accumulator. `Count` only needs the count; others need
    // sum + count for mean; min/max need a running extremum.
    #[derive(Clone, Copy)]
    struct Acc {
        sum: f64,
        count: u64,
        ext: f64, // min or max running extremum
    }
    let init = Acc { sum: 0.0, count: 0, ext: match op {
        AggOp::Min => f64::INFINITY,
        AggOp::Max => f64::NEG_INFINITY,
        _ => 0.0,
    }};

    // Preserve insertion order via a Vec<key> + HashMap<key, idx>.
    let mut key_idx: HashMap<u64, usize> = HashMap::with_capacity(keys.len() / 4);
    let mut keys_in_order: Vec<u64> = Vec::new();
    let mut accs: Vec<Acc> = Vec::new();

    for (k, v) in keys.iter().zip(values.iter()) {
        let (kk, vv) = match (k, v) {
            (Some(k), Some(v)) => (*k, *v),
            _ => continue, // skip NA key or value
        };
        let idx = *key_idx.entry(kk).or_insert_with(|| {
            keys_in_order.push(kk);
            accs.push(init);
            keys_in_order.len() - 1
        });
        let a = &mut accs[idx];
        a.sum += vv;
        a.count += 1;
        match op {
            AggOp::Min => a.ext = a.ext.min(vv),
            AggOp::Max => a.ext = a.ext.max(vv),
            _ => {}
        }
    }

    let values_out: Vec<Option<f64>> = accs.iter().map(|a| match op {
        AggOp::Sum   => Some(a.sum),
        AggOp::Mean  => if a.count > 0 { Some(a.sum / a.count as f64) } else { None },
        AggOp::Count => Some(a.count as f64),
        AggOp::Min   => if a.count > 0 { Some(a.ext) } else { None },
        AggOp::Max   => if a.count > 0 { Some(a.ext) } else { None },
    }).collect();

    HashAggResult { keys: keys_in_order, values: values_out }
}

/// Convenience: count occurrences of each unique key (R's `table()`).
/// Equivalent to `hash_agg(AggOp::Count, keys, &vec![Some(1.0); keys.len()])`
/// but skips the values traversal.
pub fn hash_tabulate(keys: &[Option<u64>]) -> HashAggResult {
    use std::collections::HashMap;
    let mut key_idx: HashMap<u64, usize> = HashMap::with_capacity(keys.len() / 4);
    let mut keys_in_order: Vec<u64> = Vec::new();
    let mut counts: Vec<u64> = Vec::new();
    for k in keys.iter().filter_map(|x| *x) {
        let idx = *key_idx.entry(k).or_insert_with(|| {
            keys_in_order.push(k);
            counts.push(0);
            keys_in_order.len() - 1
        });
        counts[idx] += 1;
    }
    HashAggResult {
        keys: keys_in_order,
        values: counts.into_iter().map(|c| Some(c as f64)).collect(),
    }
}

// ════════════════════════════════════════════════════════════════════
// Phase K.11 — Distance kernels
// ════════════════════════════════════════════════════════════════════
//
// Pairwise distance primitives shared by k-means, knn, hierarchical
// clustering, and similar pattern-detection workloads. Before K.11,
// each builtin rolled its own distance loop with the same shape but
// no parallel dispatch and no NA handling discipline. K.11 gives
// them one kernel.
//
// Distance operates on TWO same-length f64 slices (point coordinates).
// `pairwise_distance` operates on a row-major or column-major matrix
// + two row indices.
//
//   Euclidean: sqrt(sum((a_i - b_i)²))
//   Manhattan: sum(|a_i - b_i|)
//   Cosine:    1 - (a·b) / (||a|| · ||b||)
//
// NA semantics: any None in either operand at the same position
// causes that pair to be skipped (na.rm=TRUE behavior; matches R's
// `dist(..., upper=TRUE, diag=TRUE)`). If all positions have a NA on
// at least one side, returns None.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceOp {
    Euclidean,
    Manhattan,
    Cosine,
}

/// Distance between two same-length f64 slices. `None` if both slices
/// have NA at every position (degenerate) or are empty.
pub fn distance(op: DistanceOp, a: &[Option<f64>], b: &[Option<f64>]) -> Option<f64> {
    assert_eq!(a.len(), b.len(), "distance: length mismatch");
    if a.is_empty() { return None; }
    let mut count = 0usize;
    match op {
        DistanceOp::Euclidean => {
            let mut ss = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                if let (Some(xv), Some(yv)) = (x, y) {
                    let d = xv - yv;
                    ss += d * d;
                    count += 1;
                }
            }
            if count == 0 { None } else { Some(ss.sqrt()) }
        }
        DistanceOp::Manhattan => {
            let mut s = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                if let (Some(xv), Some(yv)) = (x, y) {
                    s += (xv - yv).abs();
                    count += 1;
                }
            }
            if count == 0 { None } else { Some(s) }
        }
        DistanceOp::Cosine => {
            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                if let (Some(xv), Some(yv)) = (x, y) {
                    dot += xv * yv;
                    norm_a += xv * xv;
                    norm_b += yv * yv;
                    count += 1;
                }
            }
            if count == 0 || norm_a == 0.0 || norm_b == 0.0 { return None; }
            Some(1.0 - dot / (norm_a.sqrt() * norm_b.sqrt()))
        }
    }
}

/// Compute pairwise distances for all pairs in a flat column-major
/// matrix (n rows × p cols, stored as `&[Option<f64>]` of length n*p).
/// Returns an n×n distance matrix in row-major order (Vec of length n²).
/// Uses Rayon when `n >= threshold` (n² scales fast; we go parallel early).
pub fn pairwise_distance(
    op: DistanceOp,
    data: &[Option<f64>],
    nrow: usize,
    ncol: usize,
) -> Vec<Option<f64>> {
    assert_eq!(data.len(), nrow * ncol, "pairwise_distance: shape mismatch");
    // Row i of the matrix = elements data[i*ncol..(i+1)*ncol] when
    // stored row-major. For our convention (matching how k-means + knn
    // builtins typically pass data), we assume row-major.
    let row = |i: usize| -> Vec<Option<f64>> {
        data[i * ncol..(i + 1) * ncol].to_vec()
    };
    let total_work = nrow.saturating_mul(nrow).saturating_mul(ncol);
    let go_parallel = matches!(
        r2_oracle::dispatch(r2_oracle::Op::PerPointDistance, r2_oracle::Shape::nmk(nrow, nrow, ncol)),
        r2_oracle::Backend::Rayon
    );
    if go_parallel && nrow >= 16 {
        let _ = total_work;
        par_for_rayon(nrow * nrow, |idx| {
            let i = idx / nrow;
            let j = idx % nrow;
            if i == j { Some(0.0) }
            else if j < i {
                // Diagonal symmetry — we'll fill from the (i<j) cell.
                Some(0.0) // placeholder; overwritten below
            } else {
                let ri = row(i); let rj = row(j);
                distance(op, &ri, &rj)
            }
        }).into_iter()
            .enumerate()
            .map(|(idx, v)| {
                let i = idx / nrow;
                let j = idx % nrow;
                if j < i {
                    // Mirror from (j, i): same distance.
                    let rj = row(j); let ri = row(i);
                    distance(op, &rj, &ri)
                } else { v }
            })
            .collect()
    } else {
        let mut out = vec![None; nrow * nrow];
        for i in 0..nrow {
            let ri = row(i);
            out[i * nrow + i] = Some(0.0);
            for j in (i + 1)..nrow {
                let rj = row(j);
                let d = distance(op, &ri, &rj);
                out[i * nrow + j] = d;
                out[j * nrow + i] = d;
            }
        }
        out
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<Option<f64>> {
        vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]
    }

    fn data_with_na() -> Vec<Option<f64>> {
        vec![Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)]
    }

    #[test]
    fn serial_sum_correct() {
        assert_eq!(SerialBackend.reduce(ReduceOp::Sum, &data()), Some(15.0));
    }

    #[test]
    fn rayon_sum_correct() {
        assert_eq!(RayonBackend.reduce(ReduceOp::Sum, &data()), Some(15.0));
    }

    #[test]
    fn serial_mean_correct() {
        assert_eq!(SerialBackend.reduce(ReduceOp::Mean, &data()), Some(3.0));
    }

    #[test]
    fn rayon_mean_correct() {
        assert_eq!(RayonBackend.reduce(ReduceOp::Mean, &data()), Some(3.0));
    }

    #[test]
    fn na_propagates_serial() {
        assert_eq!(SerialBackend.reduce(ReduceOp::Sum, &data_with_na()), None);
        assert_eq!(SerialBackend.reduce(ReduceOp::Mean, &data_with_na()), None);
    }

    #[test]
    fn na_propagates_rayon() {
        assert_eq!(RayonBackend.reduce(ReduceOp::Sum, &data_with_na()), None);
        assert_eq!(RayonBackend.reduce(ReduceOp::Mean, &data_with_na()), None);
    }

    #[test]
    fn min_max_prod() {
        let d = data();
        assert_eq!(SerialBackend.reduce(ReduceOp::Min, &d), Some(1.0));
        assert_eq!(SerialBackend.reduce(ReduceOp::Max, &d), Some(5.0));
        assert_eq!(SerialBackend.reduce(ReduceOp::Prod, &d), Some(120.0));
        assert_eq!(RayonBackend.reduce(ReduceOp::Min, &d), Some(1.0));
        assert_eq!(RayonBackend.reduce(ReduceOp::Max, &d), Some(5.0));
        assert_eq!(RayonBackend.reduce(ReduceOp::Prod, &d), Some(120.0));
    }

    #[test]
    fn dispatcher_picks_backend() {
        // Small input → Serial, returns same answer.
        assert_eq!(reduce(ReduceOp::Sum, &data()), Some(15.0));
    }

    #[test]
    fn empty_input() {
        let empty: Vec<Option<f64>> = vec![];
        assert_eq!(SerialBackend.reduce(ReduceOp::Sum, &empty), Some(0.0));
        assert_eq!(SerialBackend.reduce(ReduceOp::Mean, &empty), None);
        assert_eq!(SerialBackend.reduce(ReduceOp::Min, &empty), None);
    }

    // ── Map kernel tests (Phase K.2) ─────────────────────────────────

    fn map_data() -> Vec<Option<f64>> {
        vec![Some(1.0), Some(4.0), None, Some(9.0), Some(16.0)]
    }

    #[test]
    fn serial_sqrt() {
        let r = SerialBackend.map(MapOp::Sqrt, &map_data());
        assert_eq!(r, vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0)]);
    }

    #[test]
    fn rayon_sqrt() {
        let r = RayonBackend.map(MapOp::Sqrt, &map_data());
        assert_eq!(r, vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0)]);
    }

    #[test]
    fn map_abs_neg_log() {
        let d: Vec<Option<f64>> = vec![Some(-2.0), Some(0.0), Some(2.0)];
        assert_eq!(SerialBackend.map(MapOp::Abs, &d),
            vec![Some(2.0), Some(0.0), Some(2.0)]);
        assert_eq!(SerialBackend.map(MapOp::Neg, &d),
            vec![Some(2.0), Some(-0.0), Some(-2.0)]);
        let lns = SerialBackend.map(MapOp::Ln, &vec![Some(1.0), Some(std::f64::consts::E)]);
        assert!((lns[0].unwrap() - 0.0).abs() < 1e-12);
        assert!((lns[1].unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn map_dispatcher_picks_backend() {
        let r = map(MapOp::Sqrt, &map_data());
        assert_eq!(r, vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0)]);
    }

    #[test]
    fn na_preserved_in_map() {
        let d: Vec<Option<f64>> = vec![Some(4.0), None, Some(9.0)];
        let r = SerialBackend.map(MapOp::Sqrt, &d);
        assert_eq!(r[1], None);
    }

    // ── Binary kernel tests (Phase K.3) ──────────────────────────────

    #[test]
    fn binary_add_serial_and_rayon_agree() {
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let b: Vec<Option<f64>> = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let s = SerialBackend.binary(BinaryOp::Add, &a, &b);
        let r = RayonBackend.binary(BinaryOp::Add, &a, &b);
        assert_eq!(s, r);
        assert_eq!(s, vec![Some(11.0), Some(22.0), Some(33.0), Some(44.0)]);
    }

    #[test]
    fn binary_div_with_na() {
        let a: Vec<Option<f64>> = vec![Some(10.0), None, Some(9.0)];
        let b: Vec<Option<f64>> = vec![Some(2.0), Some(3.0), Some(3.0)];
        let r = SerialBackend.binary(BinaryOp::Div, &a, &b);
        assert_eq!(r, vec![Some(5.0), None, Some(3.0)]);
    }

    #[test]
    fn binary_dispatcher_picks_backend() {
        let a: Vec<Option<f64>> = vec![Some(2.0); 5];
        let b: Vec<Option<f64>> = vec![Some(3.0); 5];
        let r = binary(BinaryOp::Mul, &a, &b);
        assert_eq!(r, vec![Some(6.0); 5]);
    }

    // ── par_for tests (Phase K.4) ────────────────────────────────────

    #[test]
    fn par_for_serial_path() {
        // Small n → Oracle returns Serial → in-order indexed result.
        let r: Vec<usize> = par_for(Op::PerElementMap, 5, |i| i * 2);
        assert_eq!(r, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn par_for_rayon_path() {
        // Use TreeBuild op (threshold=1 → always Rayon-eligible) to force the
        // parallel branch. Result must still be in stable index order.
        let r: Vec<usize> = par_for(Op::TreeBuild, 100, |i| i * i);
        for (i, v) in r.iter().enumerate() {
            assert_eq!(*v, i * i);
        }
    }

    #[test]
    fn par_for_empty() {
        let r: Vec<usize> = par_for(Op::PerElementMap, 0, |i| i);
        assert!(r.is_empty());
    }

    // ── Ternary kernel tests (Phase K.5) ─────────────────────────────

    #[test]
    fn ternary_muladd_serial_and_rayon_agree() {
        // a*b + c
        let a: Vec<Option<f64>> = vec![Some(2.0), Some(3.0), Some(4.0), Some(5.0)];
        let b: Vec<Option<f64>> = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let c: Vec<Option<f64>> = vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0)];
        let s = SerialBackend.ternary(TernaryOp::MulAdd, &a, &b, &c);
        let r = RayonBackend.ternary(TernaryOp::MulAdd, &a, &b, &c);
        assert_eq!(s, r);
        assert_eq!(s, vec![Some(21.0), Some(61.0), Some(121.0), Some(201.0)]);
    }

    #[test]
    fn ternary_muladd_na_propagates_from_any_input() {
        let a: Vec<Option<f64>> = vec![Some(2.0), None, Some(4.0), Some(5.0)];
        let b: Vec<Option<f64>> = vec![Some(10.0), Some(20.0), None, Some(40.0)];
        let c: Vec<Option<f64>> = vec![Some(1.0), Some(1.0), Some(1.0), None];
        let r = SerialBackend.ternary(TernaryOp::MulAdd, &a, &b, &c);
        assert_eq!(r, vec![Some(21.0), None, None, None]);
    }

    #[test]
    fn ternary_dispatcher_routes() {
        // For a small N, dispatcher should pick Serial — but either is correct.
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(4.0), Some(5.0), Some(6.0)];
        let c: Vec<Option<f64>> = vec![Some(7.0), Some(8.0), Some(9.0)];
        let r = ternary(TernaryOp::MulAdd, &a, &b, &c);
        // 1*4+7=11, 2*5+8=18, 3*6+9=27
        assert_eq!(r, vec![Some(11.0), Some(18.0), Some(27.0)]);
    }

    // ── Strided reduction tests (Phase K.6) ──────────────────────────

    fn strided_data() -> Vec<Option<f64>> {
        // 5x3 column-major: cols are [1..5], [10..50 step 10], [100..500 step 100].
        // data[row + col*5] for column-major access.
        let mut d = Vec::with_capacity(15);
        for col_factor in [1.0, 10.0, 100.0] {
            for row in 1..=5 { d.push(Some(row as f64 * col_factor)); }
        }
        d
    }

    #[test]
    fn strided_reduce_sum_row_of_5x3() {
        let d = strided_data();
        // Row 0 (zero-indexed): elements at offset 0, stride 5 (nrow), count 3.
        // Values: 1.0, 10.0, 100.0 → sum = 111.0.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 0, 5, 3);
        assert_eq!(s, Some(111.0));
        // Row 2: offset 2, stride 5, count 3 → 3.0, 30.0, 300.0 = 333.0.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 2, 5, 3);
        assert_eq!(s, Some(333.0));
    }

    #[test]
    fn strided_reduce_serial_and_rayon_agree() {
        let d = strided_data();
        for op in [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Min, ReduceOp::Max, ReduceOp::Prod] {
            for offset in 0..5 {
                let s = SerialBackend.reduce_strided(op, &d, offset, 5, 3);
                let r = RayonBackend.reduce_strided(op, &d, offset, 5, 3);
                assert_eq!(s, r, "op={:?} offset={}", op, offset);
            }
        }
    }

    #[test]
    fn strided_reduce_na_propagates() {
        let mut d = strided_data();
        d[5] = None; // row 0 col 1 = NA → walking row 0 hits this.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 0, 5, 3);
        assert_eq!(s, None);
        // Row 1 walks indices 1, 6, 11 — does not include index 5.
        let s = SerialBackend.reduce_strided(ReduceOp::Sum, &d, 1, 5, 3);
        assert_eq!(s, Some(2.0 + 20.0 + 200.0));
    }

    #[test]
    fn strided_reduce_matches_naive_copy() {
        // Sanity: strided result == reduce over copy
        let d = strided_data();
        for offset in 0..5 {
            let copied: Vec<Option<f64>> = (0..3).map(|k| d[offset + k * 5]).collect();
            let strided = reduce_strided(ReduceOp::Mean, &d, offset, 5, 3);
            let naive = SerialBackend.reduce(ReduceOp::Mean, &copied);
            assert_eq!(strided, naive, "offset {}", offset);
        }
    }

    #[test]
    fn strided_reduce_var_and_sd() {
        let d = strided_data();
        // Row 0: values 1, 10, 100. mean=37, var=sum((x-37)^2)/(3-1).
        let var = SerialBackend.reduce_strided(ReduceOp::Var, &d, 0, 5, 3).unwrap();
        let sd  = SerialBackend.reduce_strided(ReduceOp::Sd,  &d, 0, 5, 3).unwrap();
        let mean: f64 = 37.0;
        let expected_var = ((1.0_f64 - mean).powi(2) + (10.0_f64 - mean).powi(2) + (100.0_f64 - mean).powi(2)) / 2.0;
        assert!((var - expected_var).abs() < 1e-9);
        assert!((sd - expected_var.sqrt()).abs() < 1e-9);
    }

    // ── Scan kernel tests (Phase K.7) ────────────────────────────────

    #[test]
    fn scan_cumsum_basic() {
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let r = SerialBackend.scan(ScanOp::Cumsum, &d);
        assert_eq!(r, vec![Some(1.0), Some(3.0), Some(6.0), Some(10.0)]);
    }

    #[test]
    fn scan_cumprod_basic() {
        let d: Vec<Option<f64>> = vec![Some(2.0), Some(3.0), Some(4.0)];
        let r = SerialBackend.scan(ScanOp::Cumprod, &d);
        assert_eq!(r, vec![Some(2.0), Some(6.0), Some(24.0)]);
    }

    #[test]
    fn scan_cummax_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0)];
        let r = SerialBackend.scan(ScanOp::Cummax, &d);
        assert_eq!(r, vec![Some(3.0), Some(3.0), Some(4.0), Some(4.0), Some(5.0)]);
    }

    #[test]
    fn scan_cummin_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0)];
        let r = SerialBackend.scan(ScanOp::Cummin, &d);
        assert_eq!(r, vec![Some(3.0), Some(1.0), Some(1.0), Some(1.0), Some(1.0)]);
    }

    #[test]
    fn scan_na_propagates_forward() {
        // R semantics: cumsum(c(1, 2, NA, 4)) → c(1, 3, NA, NA)
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), None, Some(4.0)];
        let r = SerialBackend.scan(ScanOp::Cumsum, &d);
        assert_eq!(r, vec![Some(1.0), Some(3.0), None, None]);
    }

    #[test]
    fn scan_serial_and_rayon_agree_on_large_input() {
        // n ≥ 4096 triggers Rayon's chunked path; verify it matches serial.
        // Use relative tolerance: cumprod accumulates to ~1e100 over 10K
        // elements, and floating-point addition / multiplication is
        // non-associative, so the chunked order produces ~ULP-level
        // differences from the sequential order. That's correct
        // behavior, not a bug.
        let n = 10_000;
        let d: Vec<Option<f64>> = (0..n).map(|i| Some(((i % 17) as f64) * 0.01 + 1.0)).collect();
        for op in [ScanOp::Cumsum, ScanOp::Cumprod, ScanOp::Cummax, ScanOp::Cummin] {
            let s = SerialBackend.scan(op, &d);
            let r = RayonBackend.scan(op, &d);
            assert_eq!(s.len(), r.len(), "op={:?}", op);
            for (i, (a, b)) in s.iter().zip(r.iter()).enumerate() {
                match (a, b) {
                    (Some(x), Some(y)) => {
                        // Skip when both overflowed to ±inf (cumprod on
                        // ≥ 1.x values past ~9K elements crosses f64::MAX).
                        if x.is_infinite() && y.is_infinite() && x.signum() == y.signum() {
                            continue;
                        }
                        let mag = x.abs().max(y.abs()).max(1.0);
                        let rel = (x - y).abs() / mag;
                        assert!(rel < 1e-12, "op={:?} i={} rel={} ({} vs {})",
                            op, i, rel, x, y);
                    }
                    (None, None) => {}
                    _ => panic!("NA mismatch op={:?} i={}: {:?} vs {:?}", op, i, a, b),
                }
            }
        }
    }

    // ── Distance kernel tests (Phase K.11) ───────────────────────────

    #[test]
    fn distance_euclidean_basic() {
        let a: Vec<Option<f64>> = vec![Some(0.0), Some(0.0), Some(0.0)];
        let b: Vec<Option<f64>> = vec![Some(3.0), Some(4.0), Some(0.0)];
        // 3-4-5 triangle in xy plane
        assert!((distance(DistanceOp::Euclidean, &a, &b).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn distance_manhattan_basic() {
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(4.0), Some(6.0), Some(3.0)];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert_eq!(distance(DistanceOp::Manhattan, &a, &b), Some(7.0));
    }

    #[test]
    fn distance_cosine_basic() {
        // Parallel vectors have cosine distance 0.
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(2.0), Some(4.0), Some(6.0)];
        let d = distance(DistanceOp::Cosine, &a, &b).unwrap();
        assert!(d.abs() < 1e-12, "cosine of parallel = {}", d);
        // Orthogonal vectors have cosine distance 1.
        let a: Vec<Option<f64>> = vec![Some(1.0), Some(0.0)];
        let b: Vec<Option<f64>> = vec![Some(0.0), Some(1.0)];
        let d = distance(DistanceOp::Cosine, &a, &b).unwrap();
        assert!((d - 1.0).abs() < 1e-12, "cosine of orthogonal = {}", d);
    }

    #[test]
    fn distance_skips_na_positions() {
        let a: Vec<Option<f64>> = vec![Some(1.0), None, Some(3.0)];
        let b: Vec<Option<f64>> = vec![Some(4.0), Some(5.0), Some(3.0)];
        // Manhattan: |1-4|=3 at idx 0; idx 1 skipped; |3-3|=0 at idx 2. Total = 3.
        assert_eq!(distance(DistanceOp::Manhattan, &a, &b), Some(3.0));
    }

    #[test]
    fn pairwise_distance_diagonal_and_symmetric() {
        // 3 points in 2D, row-major.
        let data: Vec<Option<f64>> = vec![
            Some(0.0), Some(0.0),     // point 0
            Some(3.0), Some(4.0),     // point 1
            Some(6.0), Some(8.0),     // point 2
        ];
        let d = pairwise_distance(DistanceOp::Euclidean, &data, 3, 2);
        // Diagonal should be 0.
        for i in 0..3 { assert_eq!(d[i * 3 + i], Some(0.0)); }
        // (0,1) and (1,0) should both be 5 (3-4-5 triangle).
        assert!((d[0 * 3 + 1].unwrap() - 5.0).abs() < 1e-12);
        assert!((d[1 * 3 + 0].unwrap() - 5.0).abs() < 1e-12);
        // (0,2): sqrt(36+64) = 10
        assert!((d[0 * 3 + 2].unwrap() - 10.0).abs() < 1e-12);
    }

    // ── Hash-agg kernel tests (Phase K.10) ───────────────────────────

    #[test]
    fn hash_agg_sum_basic() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(2), Some(1), Some(2), Some(3)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0), Some(50.0)];
        let r = hash_agg(AggOp::Sum, &keys, &vals);
        // Groups: key 1 → 10+30=40; key 2 → 20+40=60; key 3 → 50.
        // Insertion order: 1, 2, 3.
        assert_eq!(r.keys, vec![1, 2, 3]);
        assert_eq!(r.values, vec![Some(40.0), Some(60.0), Some(50.0)]);
    }

    #[test]
    fn hash_agg_mean_basic() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(2), Some(1), Some(2)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let r = hash_agg(AggOp::Mean, &keys, &vals);
        assert_eq!(r.values, vec![Some(20.0), Some(30.0)]);
    }

    #[test]
    fn hash_agg_count_skips_na() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(2), None, Some(1), Some(2)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(10.0), None, Some(30.0), Some(40.0), Some(50.0)];
        let r = hash_agg(AggOp::Count, &keys, &vals);
        // NA-keyed value (30.0) skipped; NA value for key=2 skipped.
        // Key 1: 2 values, Key 2: 1 value.
        assert_eq!(r.keys, vec![1, 2]);
        assert_eq!(r.values, vec![Some(2.0), Some(1.0)]);
    }

    #[test]
    fn hash_agg_min_max() {
        let keys: Vec<Option<u64>>   = vec![Some(1), Some(1), Some(1), Some(2)].into_iter().collect();
        let vals: Vec<Option<f64>>   = vec![Some(5.0), Some(2.0), Some(8.0), Some(7.0)];
        let r_min = hash_agg(AggOp::Min, &keys, &vals);
        let r_max = hash_agg(AggOp::Max, &keys, &vals);
        assert_eq!(r_min.values, vec![Some(2.0), Some(7.0)]);
        assert_eq!(r_max.values, vec![Some(8.0), Some(7.0)]);
    }

    #[test]
    fn hash_tabulate_basic() {
        let keys: Vec<Option<u64>> = vec![Some(1), Some(2), Some(1), Some(3), Some(1), Some(2)].into_iter().collect();
        let r = hash_tabulate(&keys);
        // Insertion order: 1, 2, 3 with counts 3, 2, 1.
        assert_eq!(r.keys, vec![1, 2, 3]);
        assert_eq!(r.values, vec![Some(3.0), Some(2.0), Some(1.0)]);
    }

    // ── Rolling kernel tests (Phase K.9) ─────────────────────────────

    #[test]
    fn rolling_sum_and_mean_basic() {
        // rollsum([1,2,3,4,5], 3) = [6, 9, 12]
        let d: Vec<Option<f64>> = (1..=5).map(|i| Some(i as f64)).collect();
        let r = rolling(RollingOp::Sum, &d, 3);
        assert_eq!(r, vec![Some(6.0), Some(9.0), Some(12.0)]);
        let m = rolling(RollingOp::Mean, &d, 3);
        assert_eq!(m, vec![Some(2.0), Some(3.0), Some(4.0)]);
    }

    #[test]
    fn rolling_max_min_basic() {
        let d: Vec<Option<f64>> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
            .into_iter().map(Some).collect();
        // window=3, rollmax: max(3,1,4)=4, max(1,4,1)=4, max(4,1,5)=5,
        //                    max(1,5,9)=9, max(5,9,2)=9, max(9,2,6)=9
        assert_eq!(rolling(RollingOp::Max, &d, 3),
            vec![Some(4.0), Some(4.0), Some(5.0), Some(9.0), Some(9.0), Some(9.0)]);
        assert_eq!(rolling(RollingOp::Min, &d, 3),
            vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0), Some(2.0), Some(2.0)]);
    }

    #[test]
    fn rolling_na_propagates_within_window() {
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), None, Some(4.0), Some(5.0)];
        // window=2: [1,2]=3, [2,NA]=NA, [NA,4]=NA, [4,5]=9
        assert_eq!(rolling(RollingOp::Sum, &d, 2),
            vec![Some(3.0), None, None, Some(9.0)]);
    }

    #[test]
    fn rolling_sd_basic() {
        // Window of 3 over [1,2,3,4,5]: each window has var = 1, sd = 1.
        let d: Vec<Option<f64>> = (1..=5).map(|i| Some(i as f64)).collect();
        let r = rolling(RollingOp::Sd, &d, 3);
        for v in &r {
            let x = v.unwrap();
            assert!((x - 1.0).abs() < 1e-12, "got {}", x);
        }
    }

    #[test]
    fn rolling_window_larger_than_data_returns_empty() {
        let d: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        assert!(rolling(RollingOp::Sum, &d, 10).is_empty());
        assert!(rolling(RollingOp::Sum, &d, 0).is_empty());
    }

    // ── Select kernel tests (Phase K.8) ──────────────────────────────

    #[test]
    fn select_which_max_min() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0), Some(9.0), Some(2.0)];
        assert_eq!(which_max(&d), Some(5)); // value 9.0 at index 5
        assert_eq!(which_min(&d), Some(1)); // value 1.0 at index 1 (first)
    }

    #[test]
    fn select_which_with_na() {
        let d: Vec<Option<f64>> = vec![Some(3.0), None, Some(4.0)];
        assert_eq!(which_max(&d), None);
        assert_eq!(which_min(&d), None);
    }

    #[test]
    fn select_nth_smallest_basic() {
        let d: Vec<Option<f64>> = vec![Some(5.0), Some(2.0), Some(8.0), Some(1.0), Some(3.0)];
        assert_eq!(nth_smallest(&d, 0), Some(1.0)); // min
        assert_eq!(nth_smallest(&d, 2), Some(3.0)); // median (of 5)
        assert_eq!(nth_smallest(&d, 4), Some(8.0)); // max
        assert_eq!(nth_smallest(&d, 5), None);      // out of range
    }

    #[test]
    fn select_nth_smallest_skips_na() {
        let d: Vec<Option<f64>> = vec![Some(5.0), None, Some(2.0), None, Some(8.0)];
        assert_eq!(nth_smallest(&d, 0), Some(2.0));
        assert_eq!(nth_smallest(&d, 2), Some(8.0));
        assert_eq!(nth_smallest(&d, 3), None);
    }

    #[test]
    fn select_top_k_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0), Some(9.0), Some(2.0)];
        // Top 3 in descending order of value: 9, 5, 4 → indices 5, 4, 2
        assert_eq!(top_k(&d, 3), vec![5, 4, 2]);
    }

    #[test]
    fn select_bottom_k_basic() {
        let d: Vec<Option<f64>> = vec![Some(3.0), Some(1.0), Some(4.0), Some(1.0), Some(5.0), Some(9.0), Some(2.0)];
        // Bottom 3 in ascending order: 1, 1, 2 → indices 1, 3, 6
        // (1.0 appears at both 1 and 3; tie-break by index)
        assert_eq!(bottom_k(&d, 3), vec![1, 3, 6]);
    }

    #[test]
    fn select_top_k_skips_na_and_handles_k_larger_than_data() {
        let d: Vec<Option<f64>> = vec![Some(3.0), None, Some(5.0)];
        assert_eq!(top_k(&d, 5), vec![2, 0]); // only 2 non-NA values
        assert_eq!(top_k(&d, 0), Vec::<usize>::new());
    }

    #[test]
    fn scan_rayon_handles_na_correctly() {
        // NA in the middle of a Rayon-sized chunk should poison from
        // that index forward across all subsequent chunks.
        let n = 10_000;
        let mut d: Vec<Option<f64>> = (0..n).map(|i| Some((i + 1) as f64)).collect();
        d[5000] = None;
        let s = SerialBackend.scan(ScanOp::Cumsum, &d);
        let r = RayonBackend.scan(ScanOp::Cumsum, &d);
        for i in 0..n {
            assert_eq!(s[i].is_some(), r[i].is_some(),
                "NA presence mismatch at i={}", i);
        }
        // Everything at and after i=5000 should be None.
        for i in 5000..n {
            assert!(s[i].is_none() && r[i].is_none(), "i={}", i);
        }
    }
}
