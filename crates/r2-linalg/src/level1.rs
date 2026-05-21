//! Level 1 BLAS: vector-vector operations
//! Time complexity: O(n)

/// Dot product: x · y = Σ x[i] * y[i]
/// Uses unrolled loop for better SIMD auto-vectorization.
#[inline]
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "ddot: length mismatch");
    let n = x.len();

    // 4-way unroll for SIMD
    let mut s0 = 0.0f64; let mut s1 = 0.0f64;
    let mut s2 = 0.0f64; let mut s3 = 0.0f64;
    let main = n - (n % 4);
    let mut i = 0;
    while i < main {
        s0 += x[i]     * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
        i += 4;
    }
    let mut tail = 0.0;
    while i < n { tail += x[i] * y[i]; i += 1; }
    s0 + s1 + s2 + s3 + tail
}

/// AXPY: y = α·x + y (update y in place)
#[inline]
pub fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "daxpy: length mismatch");
    if alpha == 0.0 { return; }
    let n = x.len();
    let main = n - (n % 4);
    let mut i = 0;
    while i < main {
        y[i]     += alpha * x[i];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        i += 4;
    }
    while i < n { y[i] += alpha * x[i]; i += 1; }
}

/// Scale vector: x = α·x
#[inline]
pub fn dscal(alpha: f64, x: &mut [f64]) {
    if alpha == 1.0 { return; }
    for v in x.iter_mut() { *v *= alpha; }
}

/// Euclidean norm: ||x||₂ = sqrt(x · x)
#[inline]
pub fn dnrm2(x: &[f64]) -> f64 {
    // Use scaled algorithm to avoid overflow/underflow
    let mut max: f64 = 0.0;
    for &v in x { let av = v.abs(); if av > max { max = av; } }
    if max == 0.0 { return 0.0; }
    let mut sum = 0.0;
    for &v in x { let r = v / max; sum += r * r; }
    max * sum.sqrt()
}

/// Sum of absolute values: ||x||₁
#[inline]
pub fn dasum(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

/// Index of maximum absolute value
#[inline]
pub fn idamax(x: &[f64]) -> usize {
    let mut idx = 0;
    let mut max = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        let av = v.abs();
        if av > max { max = av; idx = i; }
    }
    idx
}

/// Vector copy: y = x
#[inline]
pub fn dcopy(x: &[f64], y: &mut [f64]) {
    y.copy_from_slice(x);
}

/// Vector swap: x ↔ y
#[inline]
pub fn dswap(x: &mut [f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "dswap: length mismatch");
    for i in 0..x.len() { std::mem::swap(&mut x[i], &mut y[i]); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddot() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(ddot(&x, &y), 2.0 + 6.0 + 12.0 + 20.0);
    }

    #[test]
    fn test_daxpy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        daxpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_dnrm2() {
        let x = vec![3.0, 4.0];
        assert_eq!(dnrm2(&x), 5.0);
    }
}
