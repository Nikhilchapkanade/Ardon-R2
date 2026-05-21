// Quick Rust probe: confirm scratch pool stats show hit pattern.
use r2_memory::*;

fn main() {
    scratch_clear();
    println!("Initial pool: {} buffers, stats {:?}", scratch_pool_size(), scratch_stats());

    // First acquire — miss
    let b1 = scratch_acquire(1_000_000);
    println!("After 1st acquire: stats {:?}", scratch_stats());
    scratch_release(b1);
    println!("After release: {} buffers", scratch_pool_size());

    // Second acquire of same size — hit
    let b2 = scratch_acquire(1_000_000);
    println!("After 2nd acquire (same size): stats {:?}", scratch_stats());
    scratch_release(b2);

    // 10 acquire/release cycles of same size
    for _ in 0..10 {
        let b = scratch_acquire(1_000_000);
        scratch_release(b);
    }
    println!("After 10 cycles: stats {:?}", scratch_stats());
}
