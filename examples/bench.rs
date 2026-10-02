//! Micro-benchmarks: `cargo run --release --example bench`.
//!
//! Everything before `budgeted_cleanup` uses only the 0.1.0 API, for before/after comparisons.

use std::sync::Arc;
use std::time::{Duration, Instant};
use svcache::{CacheKey, SvCache};

#[derive(Clone)]
struct Session {
    id: u64,
    slug: String,
    _payload: [u64; 4],
}

impl CacheKey for Session {
    type Id = u64;
    fn id(&self) -> u64 {
        self.id
    }
    fn slug(&self) -> Option<&str> {
        Some(&self.slug)
    }
}

fn session(id: u64) -> Session {
    Session {
        id,
        slug: format!("s-{id}"),
        _payload: [id; 4],
    }
}

const CAP: usize = 16_384;
const THREADS: usize = 8;

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn concurrent_get() {
    let cache = Arc::new(SvCache::with_ttl_and_limit(Duration::from_secs(60), CAP));
    for i in 0..CAP as u64 {
        cache.insert(session(i));
    }
    let per_thread = 2_000_000u64;
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1);
                let mut hits = 0u64;
                for _ in 0..per_thread {
                    let id = xorshift(&mut x) % CAP as u64;
                    hits += cache.get_by_id(id).is_some() as u64;
                }
                hits
            })
        })
        .collect();
    let hits: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let el = start.elapsed();
    let ops = per_thread * THREADS as u64;
    println!(
        "concurrent get_by_id ({THREADS} thr, {CAP} entries): {:.1} ns/op/thread, {:.1} Mops/s, hit rate {:.3}",
        el.as_nanos() as f64 * THREADS as f64 / ops as f64,
        ops as f64 / el.as_secs_f64() / 1e6,
        hits as f64 / ops as f64
    );
}

fn insert_at_capacity() {
    let cache = SvCache::with_ttl_and_limit(Duration::from_secs(60), CAP);
    for i in 0..CAP as u64 {
        cache.insert(session(i));
    }
    let n = 200_000u64;
    let mut max = Duration::ZERO;
    let start = Instant::now();
    for i in 0..n {
        let t = Instant::now();
        cache.insert(session(CAP as u64 + i));
        max = max.max(t.elapsed());
    }
    let el = start.elapsed();
    println!(
        "insert at capacity (1 thr, cap {CAP}): {:.1} ns/op, max {:.1} us, len {}",
        el.as_nanos() as f64 / n as f64,
        max.as_nanos() as f64 / 1e3,
        cache.len()
    );
}

fn insert_at_capacity_mt() {
    let cache = Arc::new(SvCache::with_ttl_and_limit(Duration::from_secs(60), CAP));
    for i in 0..CAP as u64 {
        cache.insert(session(i));
    }
    let per_thread = 50_000u64;
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS as u64)
        .map(|t| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                for i in 0..per_thread {
                    cache.insert(session(1_000_000 * (t + 1) + i));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    let el = start.elapsed();
    let ops = per_thread * THREADS as u64;
    println!(
        "insert at capacity ({THREADS} thr, cap {CAP}): {:.1} Mops/s, len {}",
        ops as f64 / el.as_secs_f64() / 1e6,
        cache.len()
    );
}

fn cleanup_latency() {
    let ttl = Duration::from_millis(200);
    let cache = SvCache::with_ttl_and_limit(ttl, CAP);
    for i in 0..CAP as u64 {
        cache.insert(session(i));
    }
    std::thread::sleep(ttl + Duration::from_millis(50));
    // Gets on a live key while the whole table has expired: measures the worst single
    // call that pays for inline cleanup.
    cache.insert(session(u64::MAX));
    let n = 20_000;
    let mut lat: Vec<Duration> = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        std::hint::black_box(cache.get_by_id(u64::MAX));
        lat.push(t.elapsed());
    }
    lat.sort();
    let p = |q: f64| lat[((n as f64 - 1.0) * q) as usize].as_nanos() as f64 / 1e3;
    println!(
        "get latency while {CAP} entries expire: p50 {:.2} us, p99 {:.2} us, p99.9 {:.2} us, max {:.1} us",
        p(0.5),
        p(0.99),
        p(0.999),
        p(1.0)
    );

    let mut runs: Vec<f64> = (0..5)
        .map(|_| {
            let cache = SvCache::with_ttl_and_limit(ttl, CAP);
            for i in 0..CAP as u64 {
                cache.insert(session(i));
            }
            std::thread::sleep(ttl + Duration::from_millis(50));
            let t = Instant::now();
            assert_eq!(cache.evict_expired(), CAP);
            t.elapsed().as_nanos() as f64 / 1e3
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    println!(
        "evict_expired() full sweep of {CAP} expired entries: median {:.1} us, min {:.1} us",
        runs[2], runs[0]
    );
}

fn budgeted_cleanup() {
    let ttl = Duration::from_millis(200);
    let cache = SvCache::<Session>::builder()
        .ttl(ttl)
        .max_entries(CAP)
        .cleanup_budget(0)
        .build();
    for i in 0..CAP as u64 {
        cache.insert(session(i));
    }
    std::thread::sleep(ttl + Duration::from_millis(50));
    let mut lat = Vec::new();
    while !cache.is_empty() {
        let t = Instant::now();
        cache.evict_expired_budget(256);
        lat.push(t.elapsed().as_nanos() as f64 / 1e3);
    }
    let mean = lat.iter().sum::<f64>() / lat.len() as f64;
    lat.sort_by(f64::total_cmp);
    println!(
        "evict_expired_budget(256) x{} over {CAP} expired: mean {mean:.1} us, p50 {:.1} us, max {:.1} us",
        lat.len(),
        lat[lat.len() / 2],
        lat[lat.len() - 1]
    );
}

fn concurrent_get_with() {
    let cache = Arc::new(SvCache::with_ttl_and_limit(Duration::from_secs(60), CAP));
    for i in 0..CAP as u64 {
        cache.insert(session(i));
    }
    let per_thread = 2_000_000u64;
    let start = Instant::now();
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let cache = cache.clone();
            std::thread::spawn(move || {
                let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1);
                let mut sum = 0u64;
                for _ in 0..per_thread {
                    let id = xorshift(&mut x) % CAP as u64;
                    sum += cache.get_with(&id, |s| s._payload[0]).unwrap_or(0);
                }
                sum
            })
        })
        .collect();
    let _: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let el = start.elapsed();
    let ops = per_thread * THREADS as u64;
    println!(
        "concurrent get_with ({THREADS} thr, {CAP} entries): {:.1} ns/op/thread, {:.1} Mops/s",
        el.as_nanos() as f64 * THREADS as f64 / ops as f64,
        ops as f64 / el.as_secs_f64() / 1e6
    );
}

fn main() {
    concurrent_get();
    insert_at_capacity();
    insert_at_capacity_mt();
    cleanup_latency();
    budgeted_cleanup();
    concurrent_get_with();
}
