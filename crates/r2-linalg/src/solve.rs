//! Linear system solvers
//! Built on top of decompositions in `decomp.rs`.

use crate::{dgetrf, dpotrf, dtrsv_lower, dtrsv_upper, LinalgError};

/// Solve A·x = b using LU with partial pivoting
/// A is n×n, b is the right-hand side (consumed and replaced with solution)
pub fn dgesv(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }

    // Factorize A = P·L·U
    let ipiv = dgetrf(n, a)?;

    // Apply permutation to b
    let mut b_perm = vec![0.0; n];
    for i in 0..n { b_perm[i] = b[ipiv[i]]; }
    b.copy_from_slice(&b_perm);

    // Solve L·y = P·b (unit lower triangular, forward sub)
    dtrsv_lower(n, a, b, true)?;
    // Solve U·x = y (upper triangular, back sub)
    dtrsv_upper(n, a, b)?;
    Ok(())
}

/// Solve A·x = b where A is symmetric positive-definite (via Cholesky)
/// Much faster than dgesv when applicable (X'X in least squares)
pub fn dposv(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<(), LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }
    if b.len() != n { return Err(LinalgError::DimensionMismatch { expected: (n, 1), got: (b.len(), 1) }); }

    // Factorize A = L·Lᵀ
    dpotrf(n, a)?;
    // Solve L·y = b
    dtrsv_lower(n, a, b, false)?;
    // Solve Lᵀ·x = y — need upper triangular solve using Lᵀ
    // Since A now stores L (lower), we do back-substitution using L transposed
    // L is in column-major lower triangle of a
    dtrsv_lt(n, a, b)?;
    Ok(())
}

/// Solve Lᵀ·x = b (upper triangular back-sub using lower-stored L)
fn dtrsv_lt(n: usize, l: &[f64], b: &mut [f64]) -> Result<(), LinalgError> {
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n { sum -= l[i * n + j] * b[j]; }
        let diag = l[i * n + i];
        if diag.abs() < 1e-15 { return Err(LinalgError::Singular); }
        b[i] = sum / diag;
    }
    Ok(())
}

/// Invert a general matrix via LU decomposition
/// Returns A^(-1) in a new vector
pub fn dgetri(n: usize, a: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != n * n { return Err(LinalgError::NotSquare); }

    let mut inv = vec![0.0; n * n];
    let mut lu = a.to_vec();
    let ipiv = dgetrf(n, &mut lu)?;

    // Solve A·X = I column by column
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        // Apply permutation
        let mut ep = vec![0.0; n];
        for i in 0..n { ep[i] = e[ipiv[i]]; }
        // Solve L·y = P·e then U·x = y
        dtrsv_lower(n, &lu, &mut ep, true)?;
        dtrsv_upper(n, &lu, &mut ep)?;
        // Store column
        for i in 0..n { inv[col * n + i] = ep[i]; }
    }
    Ok(inv)
}

/// Least squares: minimize ||A·x - b||² using normal equations
/// For overdetermined systems (m > n)
/// Uses Cholesky on X'X (fast but less accurate than QR for ill-conditioned)
pub fn dlsq_normal(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if a.len() != m * n { return Err(LinalgError::InvalidShape("A".into())); }
    if b.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (b.len(), 1) }); }

    // Compute X'X (n×n) and X'b (n)
    let mut xtx = vec![0.0; n * n];
    crate::dcrossprod(m, n, a, &mut xtx)?;

    let mut xtb = vec![0.0; n];
    crate::dgemv_t(m, n, 1.0, a, b, 0.0, &mut xtb)?;

    // Solve (X'X) β = X'b
    dposv(n, &mut xtx, &mut xtb)?;
    Ok(xtb)
}

/// Optimized least squares for tall-skinny systems (m >> n, n small)
/// Fuses X'X and X'y into a single pass over X for better cache use
pub fn dlsq_fused(m: usize, n: usize, x: &[f64], y: &[f64]) -> Result<Vec<f64>, LinalgError> {
    if x.len() != m * n { return Err(LinalgError::InvalidShape("X".into())); }
    if y.len() != m { return Err(LinalgError::DimensionMismatch { expected: (m, 1), got: (y.len(), 1) }); }

    // Compute X'X and X'y simultaneously (one pass over X)
    let mut xtx = vec![0.0; n * n];
    let mut xty = vec![0.0; n];

    // Process 4 rows at a time for better vectorization
    let m4 = m - (m % 4);
    let mut r = 0;
    while r < m4 {
        let y0 = y[r]; let y1 = y[r+1]; let y2 = y[r+2]; let y3 = y[r+3];
        for j in 0..n {
            let xj0 = x[j * m + r]; let xj1 = x[j * m + r + 1];
            let xj2 = x[j * m + r + 2]; let xj3 = x[j * m + r + 3];
            xty[j] += xj0 * y0 + xj1 * y1 + xj2 * y2 + xj3 * y3;
            for i in j..n {
                let xi0 = x[i * m + r]; let xi1 = x[i * m + r + 1];
                let xi2 = x[i * m + r + 2]; let xi3 = x[i * m + r + 3];
                let dot = xi0 * xj0 + xi1 * xj1 + xi2 * xj2 + xi3 * xj3;
                xtx[j * n + i] += dot;
                if i != j { xtx[i * n + j] += dot; }
            }
        }
        r += 4;
    }
    // Handle remaining rows
    while r < m {
        let yr = y[r];
        for j in 0..n {
            let xjr = x[j * m + r];
            xty[j] += xjr * yr;
            for i in j..n {
                let dot = x[i * m + r] * xjr;
                xtx[j * n + i] += dot;
                if i != j { xtx[i * n + j] += dot; }
            }
        }
        r += 1;
    }

    // Solve via Cholesky
    dposv(n, &mut xtx, &mut xty)?;
    Ok(xty)
}

/// Direct 2×2 solve (avoids all overhead for simple regression)
#[inline]
pub fn solve_2x2(a: &[f64; 4], b: &[f64; 2]) -> Result<[f64; 2], LinalgError> {
    // a is column-major 2×2: a[0]=a11, a[1]=a21, a[2]=a12, a[3]=a22
    let det = a[0] * a[3] - a[1] * a[2];
    if det.abs() < 1e-15 { return Err(LinalgError::Singular); }
    let inv_det = 1.0 / det;
    Ok([
        inv_det * (a[3] * b[0] - a[2] * b[1]),
        inv_det * (a[0] * b[1] - a[1] * b[0]),
    ])
}

/// Direct 3×3 solve via Cramer's rule (for 2-predictor regression with intercept)
#[inline]
pub fn solve_3x3(a: &[f64; 9], b: &[f64; 3]) -> Result<[f64; 3], LinalgError> {
    // Column-major 3×3
    let det = a[0]*(a[4]*a[8]-a[5]*a[7]) - a[3]*(a[1]*a[8]-a[2]*a[7]) + a[6]*(a[1]*a[5]-a[2]*a[4]);
    if det.abs() < 1e-15 { return Err(LinalgError::Singular); }
    let inv = 1.0 / det;
    Ok([
        inv * (b[0]*(a[4]*a[8]-a[5]*a[7]) - a[3]*(b[1]*a[8]-a[2]*b[2]) + a[6]*(b[1]*a[5]-a[2]*b[2])),
        inv * (a[0]*(b[1]*a[8]-b[2]*a[7]) - b[0]*(a[1]*a[8]-a[2]*a[7]) + a[6]*(a[1]*b[2]-a[2]*b[1])),
        inv * (a[0]*(a[4]*b[2]-a[5]*b[1]) - a[3]*(a[1]*b[2]-a[2]*b[1]) + b[0]*(a[1]*a[5]-a[2]*a[4])),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgesv() {
        // A = [[2, 1], [1, 3]], b = [5, 10]
        // Solution: x = [1, 3]
        let mut a = vec![2.0, 1.0, 1.0, 3.0]; // column-major
        let mut b = vec![5.0, 10.0];
        dgesv(2, &mut a, &mut b).unwrap();
        assert!((b[0] - 1.0).abs() < 1e-10);
        assert!((b[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dlsq_normal() {
        // Simple regression: y = 2x + 1 with noise
        // X = [[1, 1], [1, 2], [1, 3], [1, 4]] column-major
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![3.0, 5.0, 7.0, 9.0]; // exactly 2x + 1
        let beta = dlsq_normal(4, 2, &x, &y).unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-8); // intercept
        assert!((beta[1] - 2.0).abs() < 1e-8); // slope
    }
}
