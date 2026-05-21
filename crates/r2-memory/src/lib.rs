//! R2 memory layer.
//!
//! Two distinct services live here:
//!
//! 1. **Heap-usage tracking** — `track_alloc` / `heap_used`. Used by the
//!    Oracle (and future memory-pressure backpressure logic) to know
//!    how much memory is currently committed to RVal payloads.
//!
//! 2. **`F64ScratchPool`** — a thread-local recyclable pool of
//!    `Vec<f64>` buffers. Builtins that need a short-lived temporary
//!    numeric vector (e.g. `median` materialising filtered values for
//!    quickselect, distance kernels copying into contiguous form,
//!    sort-then-reduce paths) can `scratch_acquire(n)` instead of
//!    allocating fresh. The buffer goes back to the pool via
//!    `scratch_release(buf)` when the caller is done.
//!
//!    Wins: avoids ~5-50µs of allocator overhead per call for
//!    medium-to-large buffers, and avoids zero-filling on Vec creation
//!    (the returned buffer has its capacity but `len=0`; the caller
//!    fills it explicitly).
//!
//!    Doesn't help: cases where the allocated buffer becomes long-lived
//!    output (e.g. binary op result that goes into a `Reals`) — those
//!    must keep heap-allocated lifetime.
//!
//!    Bucket sizing: powers of 2 from 16 elements (128 B) up through
//!    2^28 (~2 GB). A request for `n` items returns a buffer with
//!    capacity ≥ ceil_pow2(n). On release, the buffer goes into the
//!    matching bucket. Per-bucket cap of 4 buffers prevents unbounded
//!    growth on workloads that churn many distinct sizes.

use std::sync::atomic::{AtomicUsize, Ordering};

static HEAP_USED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskSize { Small, Medium, Large }

impl TaskSize {
    pub fn classify(n: usize) -> Self {
        if n < 1_000_000 { TaskSize::Small }
        else if n < 100_000_000 { TaskSize::Medium }
        else { TaskSize::Large }
    }
    pub fn should_parallelize(&self) -> bool { !matches!(self, TaskSize::Small) }
}

pub fn track_alloc(bytes: usize) { HEAP_USED.fetch_add(bytes, Ordering::Relaxed); }
pub fn heap_used() -> usize { HEAP_USED.load(Ordering::Relaxed) }

// ════════════════════════════════════════════════════════════════════
// F64ScratchPool — per-thread recyclable `Vec<f64>` buffers
// ════════════════════════════════════════════════════════════════════

/// Maximum bucket index (covers up to 2^MAX_BUCKET = 256M elements).
const MAX_BUCKET: usize = 28;
/// Per-bucket cap — prevents unbounded growth when workloads churn
/// many distinct sizes. 4 is a deliberate small number; the pool's
/// value is per-iteration recycling within a single workload, not
/// hoarding every distinct size we ever saw.
const PER_BUCKET_CAP: usize = 4;

/// Compute the bucket index for a given capacity request.
/// Returns the smallest index `b` such that `2^b >= cap`.
#[inline]
fn bucket_for(cap: usize) -> usize {
    if cap <= 1 { return 0; }
    // ceil(log2(cap)) = position of the highest bit after rounding up.
    let n = cap.next_power_of_two();
    n.trailing_zeros() as usize
}

#[inline]
fn bucket_capacity(bucket: usize) -> usize {
    1usize << bucket
}

/// Bounded per-thread pool. Buffers are grouped by power-of-two
/// capacity. `len()` is always 0 on returned buffers; capacity ≥
/// requested.
pub struct F64ScratchPool {
    buckets: Vec<Vec<Vec<f64>>>,
    /// Statistics (reset on `clear`).
    pub hits: usize,
    pub misses: usize,
}

impl F64ScratchPool {
    pub fn new() -> Self {
        let mut buckets = Vec::with_capacity(MAX_BUCKET + 1);
        for _ in 0..=MAX_BUCKET { buckets.push(Vec::with_capacity(PER_BUCKET_CAP)); }
        F64ScratchPool { buckets, hits: 0, misses: 0 }
    }

    /// Get a buffer with capacity ≥ `min_cap`. Returns an empty
    /// (len=0) Vec — the caller is responsible for pushing or
    /// extending it. The capacity will be at least `min_cap`.
    pub fn acquire(&mut self, min_cap: usize) -> Vec<f64> {
        let b = bucket_for(min_cap);
        if b > MAX_BUCKET {
            // Requested size too large for our buckets — fall back to
            // a fresh allocation (release will also drop it).
            self.misses += 1;
            return Vec::with_capacity(min_cap);
        }
        if let Some(mut buf) = self.buckets[b].pop() {
            self.hits += 1;
            buf.clear(); // len=0, capacity preserved
            return buf;
        }
        self.misses += 1;
        Vec::with_capacity(bucket_capacity(b))
    }

    /// Return a buffer to the pool. Caller surrenders ownership.
    /// The buffer is cleared (len=0) and stashed in its bucket.
    /// If the bucket is full, the buffer is dropped.
    pub fn release(&mut self, mut buf: Vec<f64>) {
        let cap = buf.capacity();
        let b = bucket_for(cap.max(1));
        if b > MAX_BUCKET { return; } // too big — drop
        if self.buckets[b].len() >= PER_BUCKET_CAP { return; } // bucket full — drop
        buf.clear();
        self.buckets[b].push(buf);
    }

    /// Empty the pool. Called by long-running processes that want to
    /// release pooled memory back to the system allocator.
    pub fn clear(&mut self) {
        for b in &mut self.buckets { b.clear(); }
        self.hits = 0;
        self.misses = 0;
    }

    /// Total buffers held across all buckets.
    pub fn buffer_count(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }
}

impl Default for F64ScratchPool {
    fn default() -> Self { Self::new() }
}

// ── Thread-local pool + free helpers ─────────────────────────────────

thread_local! {
    static SCRATCH_POOL: std::cell::RefCell<F64ScratchPool> =
        std::cell::RefCell::new(F64ScratchPool::new());
}

/// Acquire a scratch `Vec<f64>` from this thread's pool.
pub fn scratch_acquire(min_cap: usize) -> Vec<f64> {
    SCRATCH_POOL.with(|p| p.borrow_mut().acquire(min_cap))
}

/// Return a scratch `Vec<f64>` to this thread's pool.
pub fn scratch_release(buf: Vec<f64>) {
    SCRATCH_POOL.with(|p| p.borrow_mut().release(buf));
}

/// Run `f` with a scratch buffer, automatically releasing on return.
/// `f` can mutate the buffer; on return the buffer goes back to the
/// pool (any data in it is discarded).
pub fn with_scratch<R, F: FnOnce(&mut Vec<f64>) -> R>(min_cap: usize, f: F) -> R {
    let mut buf = scratch_acquire(min_cap);
    let r = f(&mut buf);
    scratch_release(buf);
    r
}

/// Diagnostic — current pool buffer count for this thread.
pub fn scratch_pool_size() -> usize {
    SCRATCH_POOL.with(|p| p.borrow().buffer_count())
}

/// Diagnostic — hit / miss counters for this thread's pool.
pub fn scratch_stats() -> (usize, usize) {
    SCRATCH_POOL.with(|p| {
        let pp = p.borrow();
        (pp.hits, pp.misses)
    })
}

/// Reset this thread's pool (drops all pooled buffers, zeroes stats).
pub fn scratch_clear() {
    SCRATCH_POOL.with(|p| p.borrow_mut().clear());
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_for_basic() {
        assert_eq!(bucket_for(0), 0);
        assert_eq!(bucket_for(1), 0);
        assert_eq!(bucket_for(2), 1);
        assert_eq!(bucket_for(3), 2);
        assert_eq!(bucket_for(4), 2);
        assert_eq!(bucket_for(5), 3);
        assert_eq!(bucket_for(1000), 10);   // 2^10 = 1024
        assert_eq!(bucket_for(1024), 10);
        assert_eq!(bucket_for(1025), 11);
    }

    #[test]
    fn pool_recycles_buffers() {
        let mut p = F64ScratchPool::new();
        let b1 = p.acquire(100);
        assert!(b1.capacity() >= 100);
        assert_eq!(b1.len(), 0);
        let cap1 = b1.capacity();
        p.release(b1);

        let b2 = p.acquire(100);
        // Should reuse the same buffer (same capacity).
        assert_eq!(b2.capacity(), cap1);
        assert_eq!(b2.len(), 0);
        assert_eq!(p.hits, 1);
        assert_eq!(p.misses, 1);
    }

    #[test]
    fn pool_handles_different_sizes() {
        let mut p = F64ScratchPool::new();
        let b1 = p.acquire(100);    // bucket 7 (cap 128)
        let b2 = p.acquire(1000);   // bucket 10 (cap 1024)
        let b3 = p.acquire(100_000); // bucket 17 (cap 131072)
        p.release(b1);
        p.release(b2);
        p.release(b3);
        assert_eq!(p.buffer_count(), 3);
        // Acquiring 100 should hit the small bucket.
        let _ = p.acquire(100);
        assert_eq!(p.hits, 1);
    }

    #[test]
    fn pool_respects_per_bucket_cap() {
        let mut p = F64ScratchPool::new();
        // Release 6 buffers of the same size — only PER_BUCKET_CAP (4) kept.
        for _ in 0..6 {
            p.release(Vec::with_capacity(1024));
        }
        // Bucket 10 should have 4 buffers (2 dropped).
        assert_eq!(p.buckets[10].len(), PER_BUCKET_CAP);
    }

    #[test]
    fn pool_returns_clean_len_zero() {
        let mut p = F64ScratchPool::new();
        let mut b = p.acquire(10);
        b.extend([1.0, 2.0, 3.0]);
        assert_eq!(b.len(), 3);
        p.release(b);
        let b2 = p.acquire(10);
        assert_eq!(b2.len(), 0, "released buffer should come back with len=0");
    }

    #[test]
    fn with_scratch_releases_on_drop() {
        scratch_clear();
        let initial = scratch_pool_size();
        with_scratch(1000, |buf| {
            buf.extend([1.0, 2.0, 3.0]);
            assert_eq!(buf.len(), 3);
        });
        // Buffer should be back in pool.
        assert_eq!(scratch_pool_size(), initial + 1);
    }

    #[test]
    fn thread_local_acquire_release() {
        scratch_clear();
        let b = scratch_acquire(500);
        assert!(b.capacity() >= 500);
        scratch_release(b);
        assert_eq!(scratch_pool_size(), 1);
        let (_, misses) = scratch_stats();
        assert!(misses >= 1, "first acquire is a miss");
    }
}
