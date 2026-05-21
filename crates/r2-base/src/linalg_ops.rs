//! R2 Linalg builtins — Phase R.4.
//!
//! Domain crate for builtin functions whose body is a thin wrapper over
//! the BLAS-style numerical kernels (`dgesvd`, `dsyev`, …) plus the
//! lightweight constructors that produce `RVal::Matrix` / `RVal::Tensor`.
//!
//! All builtins follow the locked pure pattern
//! `fn(&[EvalArg]) -> Result<RVal, R2Err>` — no Engine dependency. Args
//! coerce via `RVal::as_reals()` / `scalar_f64()` from r2-types.
//!
//! Honesty notes:
//!   - `bi_eigen` calls Jacobi `dsyev`. Correct, slow for n ≳ 100.
//!   - `bi_svd` returns ONLY `$d` (singular values) — it intentionally
//!     omits `$u` and `$v` because the underlying `dgesvd` does not yet
//!     accumulate the orthogonal factors. Returning identity matrices
//!     while pretending success would silently corrupt callers; we
//!     refuse the temptation. Full U/Vᵀ is on the roadmap (tridiag+QR
//!     replacement of Jacobi, see KNOWN_LIMITATIONS).

use r2_linalg::{dgesvd, dgesvd_full, dsyev, dsyev_full};
use r2_types::{fmt_num, Attrs, ErrKind, EvalArg, Matrix, R2Err, RVal, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

const MAX_ALLOC_BYTES: usize = 500_000_000;

#[inline]
fn check_alloc(elements: usize, elem_size: usize) -> Result<(), R2Err> {
    let bytes = elements.saturating_mul(elem_size);
    if bytes > MAX_ALLOC_BYTES {
        return Err(R2Err {
            msg: format!(
                "allocation of {} bytes exceeds limit (max {} MB). Use chunked processing for large data.",
                bytes,
                MAX_ALLOC_BYTES / 1_000_000
            ),
            kind: ErrKind::Runtime,
        });
    }
    Ok(())
}

#[inline]
fn gv(args: &[EvalArg], i: usize) -> RVal {
    args.get(i).map(|a| a.value.clone()).unwrap_or(RVal::Null)
}

#[inline]
fn gn(args: &[EvalArg], name: &str) -> Option<RVal> {
    args.iter()
        .find(|a| a.name.as_ref().map(|n| n.as_ref()) == Some(name))
        .map(|a| a.value.clone())
}

#[inline]
fn rnums(v: &[f64]) -> RVal {
    RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
}

// ─────────────────────────────────────────────────────────────────────
// matrix(), tensor(), t(), crossprod()
// ─────────────────────────────────────────────────────────────────────

/// `matrix(data, nrow=, ncol=)` — fill a Matrix with the given data.
pub fn bi_matrix(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let data: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    // R's matrix(data, nrow, ncol, byrow=FALSE) — nrow/ncol accept BOTH
    // positional and named forms. Previously we only honoured named args,
    // so `matrix(rnorm(1e6), 1000, 1000)` silently became 1e6×1.
    let positional_nrow = a.get(1).filter(|p| p.name.is_none())
        .and_then(|p| p.value.scalar_f64().ok().flatten());
    let positional_ncol = a.get(2).filter(|p| p.name.is_none())
        .and_then(|p| p.value.scalar_f64().ok().flatten());
    let nrow = gn(a, "nrow").and_then(|v| v.scalar_f64().ok().flatten())
        .or(positional_nrow)
        .map(|n| n as usize);
    let ncol = gn(a, "ncol").and_then(|v| v.scalar_f64().ok().flatten())
        .or(positional_ncol)
        .map(|n| n as usize);
    let byrow = gn(a, "byrow").and_then(|v| v.as_logicals().ok())
        .and_then(|v| v.first().copied().flatten()).unwrap_or(false);
    let (nr, nc) = match (nrow, ncol) {
        (Some(r), Some(c)) => (r, c),
        (Some(r), None) => (r, (data.len() + r.saturating_sub(1)) / r.max(1)),
        (None, Some(c)) => ((data.len() + c.saturating_sub(1)) / c.max(1), c),
        (None, None) => (data.len(), 1),
    };
    check_alloc(nr * nc, 8)?;
    let mut d = if byrow {
        // R's byrow=TRUE: fill row-major, store column-major (R's convention).
        let mut out = vec![0.0; nr * nc];
        for i in 0..nr {
            for j in 0..nc {
                let src = i * nc + j;
                if src < data.len() { out[j * nr + i] = data[src]; }
            }
        }
        out
    } else {
        // Column-major fill (R default): values map directly.
        let mut v = data;
        v.resize(nr * nc, 0.0);
        v
    };
    let _ = &mut d; // silence unused-mut warning when byrow branch yields vec without mutation
    Ok(RVal::Matrix(Matrix::new(d, nr, nc)))
}

/// `tensor(data, shape=)` — n-dimensional tensor constructor.
pub fn bi_tensor(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let data: Vec<f64> = gv(a, 0).as_reals()?.into_iter().filter_map(|x| x).collect();
    let shape: Vec<usize> = if let Some(s) = gn(a, "shape") {
        s.as_reals()?.into_iter().filter_map(|x| x.map(|n| n as usize)).collect()
    } else {
        vec![data.len()]
    };
    Ok(RVal::Tensor(Tensor::new(data, shape)))
}

/// `t(M)` — matrix transpose.
pub fn bi_transpose(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Matrix(m) => Ok(RVal::Matrix(m.transpose())),
        _ => Err(R2Err { msg: "t() needs matrix".into(), kind: ErrKind::Runtime }),
    }
}

/// `crossprod(M)` — Mᵀ · M.
pub fn bi_crossprod(a: &[EvalArg]) -> Result<RVal, R2Err> {
    match &gv(a, 0) {
        RVal::Matrix(m) => Ok(RVal::Matrix(m.crossprod())),
        _ => Err(R2Err { msg: "crossprod needs matrix".into(), kind: ErrKind::Runtime }),
    }
}

// ─────────────────────────────────────────────────────────────────────
// svd(), eigen()
// ─────────────────────────────────────────────────────────────────────

/// `svd(M)` — thin singular value decomposition: M = U · diag(d) · Vᵀ.
///
/// Returns a list with three fields:
///   - `$d`: n singular values in descending order
///   - `$u`: m×n column-major matrix with orthonormal columns
///   - `$v`: n×n column-major matrix with orthonormal columns (V itself,
///     matching R's convention; Vᵀ is what one multiplies on the right
///     during reconstruction).
///
/// Implementation: Householder bidiagonalization (Golub-Kahan) with the
/// orthogonal factors recovered via reverse application of the stored
/// reflectors onto thin identities (`dorgbr`-style), then diagonalization
/// of the bidiagonal B by symmetric eigendecomposition of Bᵀ·B via the
/// already-shipped `dsyev_full` (Householder tridiag + Wilkinson-shift QR).
///
/// **Honest accuracy note:** the Bᵀ·B route squares the condition number.
/// For well-conditioned matrices (κ ≲ 1e7) singular values and vectors are
/// accurate to ~1e-12. For badly conditioned matrices, small singular
/// values lose accuracy proportionally. A proper LAPACK `dbdsqr`
/// (implicit-shift bidiagonal QR with full Givens accumulation) would
/// give κ-independent accuracy at higher implementation cost — tracked
/// in KNOWN_LIMITATIONS.
pub fn bi_svd(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mat = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "svd() needs a matrix".into(), kind: ErrKind::Runtime }),
    };
    let (m, n) = (mat.nrow, mat.ncol);
    let (sigma, u_data, vt_data) = dgesvd_full(m, n, &mat.data)
        .map_err(|e| R2Err { msg: format!("svd failed: {}", e), kind: ErrKind::Runtime })?;

    // R returns V (not Vᵀ) in the $v field. Transpose Vᵀ → V.
    let mut v_data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            v_data[j * n + i] = vt_data[i * n + j];
        }
    }

    let mut fields: HashMap<Arc<str>, RVal> = HashMap::new();
    fields.insert(Arc::from("d"), rnums(&sigma));
    fields.insert(Arc::from("u"), RVal::Matrix(Matrix::new(u_data, m, n)));
    fields.insert(Arc::from("v"), RVal::Matrix(Matrix::new(v_data, n, n)));
    Ok(RVal::List(fields.into_iter().map(|(k, v)| (Some(k), v)).collect()))
}

/// `eigen(A)` — symmetric eigendecomposition with **eigenvectors** (Tier 1).
///
/// Returns `$values` (descending) and `$vectors` (column-major matrix where
/// column i is the eigenvector for `values[i]`). Implementation: Householder
/// tridiagonalization → implicit symmetric QR with Wilkinson shift → back-
/// transform. Replaces the previous Jacobi-eigenvalues-only path.
///
/// The standalone `dsyev` kernel (values-only Jacobi) is retained for
/// callers that don't need vectors and want the simpler code path.
pub fn bi_eigen(a: &[EvalArg]) -> Result<RVal, R2Err> {
    let mat = match &gv(a, 0) {
        RVal::Matrix(m) => m.clone(),
        _ => return Err(R2Err { msg: "eigen() needs a square matrix".into(), kind: ErrKind::Runtime }),
    };
    if mat.nrow != mat.ncol {
        return Err(R2Err { msg: "eigen() needs a square matrix".into(), kind: ErrKind::Runtime });
    }
    let (eigenvalues, vectors) = dsyev_full(mat.nrow, &mat.data)
        .map_err(|e| R2Err { msg: format!("eigen failed: {}", e), kind: ErrKind::Runtime })?;

    // No auto-print: R's `e <- eigen(A)` returns invisibly; only direct
    // top-level `eigen(A)` should display, handled by the REPL's
    // auto-print path on the returned RVal. Caller can `print(e)`,
    // `summary(e)`, or `e$values` / `e$vectors` to inspect.
    let mut fields: HashMap<Arc<str>, RVal> = HashMap::new();
    fields.insert(Arc::from("values"), rnums(&eigenvalues));
    fields.insert(Arc::from("vectors"), RVal::Matrix(Matrix::new(vectors, mat.nrow, mat.nrow)));
    Ok(RVal::List(fields.into_iter().map(|(k, v)| (Some(k), v)).collect()))
}

// ─────────────────────────────────────────────────────────────────────
// Builtins registry (Phase R.4).
// ─────────────────────────────────────────────────────────────────────

/// Returns this crate's exported builtins as `(name, fn-pointer)` pairs.
pub fn register_builtins() -> Vec<(&'static str, fn(&[EvalArg]) -> Result<RVal, R2Err>)> {
    vec![
        ("matrix",    bi_matrix),
        ("tensor",    bi_tensor),
        ("t",         bi_transpose),
        ("crossprod", bi_crossprod),
        ("svd",       bi_svd),
        ("eigen",     bi_eigen),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nums(v: &[f64]) -> RVal {
        RVal::Numeric(v.iter().map(|x| Some(*x)).collect(), Attrs::default())
    }
    fn evarg(v: RVal) -> EvalArg { EvalArg { name: None, value: v } }
    fn evarg_named(name: &str, v: RVal) -> EvalArg {
        EvalArg { name: Some(Arc::from(name)), value: v }
    }

    #[test]
    fn matrix_fills_correct_shape() {
        let a = vec![
            evarg(nums(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            evarg_named("nrow", nums(&[2.0])),
            evarg_named("ncol", nums(&[3.0])),
        ];
        let r = bi_matrix(&a).unwrap();
        match r {
            RVal::Matrix(m) => { assert_eq!(m.nrow, 2); assert_eq!(m.ncol, 3); }
            _ => panic!("matrix() must return Matrix"),
        }
    }

    #[test]
    fn transpose_swaps_dims() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let r = bi_transpose(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::Matrix(t) => { assert_eq!(t.nrow, 3); assert_eq!(t.ncol, 2); }
            _ => panic!("t() must return Matrix"),
        }
    }

    #[test]
    fn eigen_diagonal_returns_diag() {
        // 2x2 diag(3, 7) → eigenvalues are 3 and 7.
        let m = Matrix::new(vec![3.0, 0.0, 0.0, 7.0], 2, 2);
        let r = bi_eigen(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::List(items) => {
                let (_, vals) = items.iter().find(|(k, _)| k.as_deref().map(|s| s.as_ref()) == Some("values")).unwrap();
                let v = vals.as_reals().unwrap();
                let mut sorted: Vec<f64> = v.iter().filter_map(|x| *x).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!((sorted[0] - 3.0).abs() < 1e-10);
                assert!((sorted[1] - 7.0).abs() < 1e-10);
            }
            _ => panic!("eigen() must return List"),
        }
    }

    #[test]
    fn svd_returns_d_u_v() {
        // 3x2 with known SVs (= 4, 3 since the columns are scaled basis vectors):
        // rows = [3,0; 0,4; 0,0]; column-major data.
        let data = vec![3.0, 0.0, 0.0,  0.0, 4.0, 0.0];
        let m_in = Matrix::new(data.clone(), 3, 2);
        let r = bi_svd(&[evarg(RVal::Matrix(m_in))]).unwrap();
        match r {
            RVal::List(items) => {
                let by_name: HashMap<&str, &RVal> = items.iter()
                    .filter_map(|(k, v)| k.as_deref().map(|s| (s.as_ref(), v)))
                    .collect();
                assert!(by_name.contains_key("d"));
                assert!(by_name.contains_key("u"));
                assert!(by_name.contains_key("v"));
                // Singular values descending: [4, 3].
                let d = match by_name["d"] { RVal::Numeric(d, _) => d, _ => panic!("$d not Numeric") };
                assert!((d[0].unwrap() - 4.0).abs() < 1e-10);
                assert!((d[1].unwrap() - 3.0).abs() < 1e-10);
                // U is 3×2, V is 2×2.
                let u = match by_name["u"] { RVal::Matrix(u) => u, _ => panic!("$u not Matrix") };
                let v = match by_name["v"] { RVal::Matrix(v) => v, _ => panic!("$v not Matrix") };
                assert_eq!((u.nrow, u.ncol), (3, 2));
                assert_eq!((v.nrow, v.ncol), (2, 2));
                // Reconstruct A = U · diag(d) · Vᵀ and compare.
                for j in 0..2 {
                    for i in 0..3 {
                        let mut s = 0.0;
                        for k in 0..2 {
                            // Vᵀ[k, j] = v.data[k * 2 + j]  (V is col-major n×n, so
                            //   V[j, k] = v.data[k * 2 + j]; Vᵀ[k, j] = V[j, k])
                            s += u.data[k * 3 + i] * d[k].unwrap() * v.data[k * 2 + j];
                        }
                        let orig = data[j * 3 + i];
                        assert!((orig - s).abs() < 1e-9, "rec[{},{}]: {} vs {}", i, j, orig, s);
                    }
                }
            }
            _ => panic!("svd() must return List"),
        }
    }

    #[test]
    fn crossprod_yields_square() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let r = bi_crossprod(&[evarg(RVal::Matrix(m))]).unwrap();
        match r {
            RVal::Matrix(c) => { assert_eq!(c.nrow, 3); assert_eq!(c.ncol, 3); }
            _ => panic!("crossprod() must return Matrix"),
        }
    }
}
