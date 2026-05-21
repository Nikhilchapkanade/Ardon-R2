//! Bare-metal DGEMM micro-benchmark. Times r2_linalg::dgemm on a 500×500
//! square matmul over several iterations. Measures pure kernel throughput
//! with no R2 engine overhead, no Vec<Option<f64>> conversions.
//!
//! Run: cargo run --release --example bench_dgemm -p r2-linalg

use std::time::Instant;

fn main() {
    let n: usize = 500;
    // Initialize A and B with deterministic values
    let a: Vec<f64> = (0..n*n).map(|i| ((i % 100) as f64) * 0.01).collect();
    let b: Vec<f64> = (0..n*n).map(|i| ((i % 50)  as f64) * 0.02).collect();
    let mut c = vec![0.0_f64; n*n];

    // Warmup
    r2_linalg::dgemm(n, n, n, 1.0, &a, &b, 0.0, &mut c).unwrap();

    // Time 5 iterations
    let mut total = 0.0;
    for i in 0..5 {
        c.iter_mut().for_each(|x| *x = 0.0);
        let t0 = Instant::now();
        r2_linalg::dgemm(n, n, n, 1.0, &a, &b, 0.0, &mut c).unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        total += elapsed;
        let flops = 2.0 * (n as f64).powi(3);
        println!("iter {}: {:.4}s = {:.2} GFLOPS", i, elapsed, flops / elapsed / 1e9);
    }
    let avg = total / 5.0;
    let flops = 2.0 * (n as f64).powi(3);
    println!("\navg 5 iters: {:.4}s = {:.2} GFLOPS", avg, flops / avg / 1e9);
    println!("(R reference Rblas typically gets ~0.25 GFLOPS on this size)");
}
