//! Deterministic traces and bounded timings for comparison with a baseline revision.
//! `cargo run --release --locked --example expiry_bench`
//! Copy this unchanged example to the baseline checkout before comparing outputs.
//! Timings include only the named phase; no absolute performance thresholds are asserted.
use std::time::{Duration, Instant};
use svcache::{CacheKey, ManualClock, SvCache};

#[derive(Clone)]
struct Item(u64);
impl CacheKey for Item {
    type Id = u64;
    fn id(&self) -> u64 {
        self.0
    }
}

fn cleanup(trial: usize, sparse: bool) {
    const PEAK: u64 = 32_768;
    let clock = ManualClock::new();
    let cache = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .cleanup_budget(0)
        .clock(clock.clone())
        .build();
    for key in 0..PEAK {
        cache.insert(Item(key));
    }
    if sparse {
        for key in 0..PEAK - 64 {
            cache.remove(&key);
        }
    }
    let remaining = cache.len();
    clock.advance(Duration::from_secs(11));
    let start = Instant::now();
    let mut calls = 0;
    let mut removed = 0;
    while !cache.is_empty() {
        removed += cache.evict_expired_budget(64);
        calls += 1;
        assert!(calls <= PEAK, "cleanup failed to make progress");
    }
    let elapsed = start.elapsed().as_nanos();
    assert_eq!(removed, remaining);
    println!(
        "{trial},{},32768,{calls},{removed},{elapsed}",
        if sparse {
            "sparse_cleanup"
        } else {
            "dense_cleanup"
        }
    );
}

fn ttl_pressure(trial: usize) {
    let start = Instant::now();
    let mut hits = 0;
    for _ in 0..1_000 {
        let clock = ManualClock::new();
        let cache = SvCache::builder()
            .ttl(Duration::from_secs(10))
            .cleanup_interval(Duration::from_secs(11))
            .cleanup_budget(4)
            .max_entries(4)
            .clock(clock.clone())
            .build();
        for key in 0..3 {
            cache.insert(Item(key));
        }
        clock.advance(Duration::from_secs(8));
        assert!(cache.get_by_id(0).is_some());
        cache.insert(Item(3));
        clock.advance(Duration::from_secs(1));
        cache.insert(Item(4));
        clock.advance(Duration::from_secs(1));
        assert!(cache.touch(&2));
        clock.advance(Duration::from_secs(1));
        cache.insert(Item(5));
        hits += cache.get_by_id(3).is_some() as usize;
    }
    println!(
        "{trial},ttl_pressure_probe,4,1000,{hits},{}",
        start.elapsed().as_nanos()
    );
}

fn mixed_hits(trial: usize) {
    let cache = SvCache::with_limit(1024);
    let mut hits = 0;
    let start = Instant::now();
    for request in 0..200_000 {
        let key = if request % 10 == 0 {
            1_000_000 + request
        } else {
            request % 64
        };
        if cache.get_by_id(key).is_some() {
            hits += 1;
        } else {
            cache.insert(Item(key));
        }
    }
    println!(
        "{trial},mixed_sieve_hits,1024,200000,{hits},{}",
        start.elapsed().as_nanos()
    );
}

fn capacity_churn(trial: usize) {
    let cache = SvCache::with_limit(1024);
    for key in 0..1024 {
        cache.insert(Item(key));
    }
    let start = Instant::now();
    for key in 1024..51_024 {
        cache.insert(Item(key));
    }
    println!(
        "{trial},capacity_churn,1024,50000,{},{}",
        cache.len(),
        start.elapsed().as_nanos()
    );
    assert_eq!(cache.len(), 1024);
}

fn main() {
    println!("trial,workload,capacity_or_peak,operations,removed_or_hits_or_retained,elapsed_ns");
    for trial in 0..5 {
        cleanup(trial, true);
        cleanup(trial, false);
        ttl_pressure(trial);
        mixed_hits(trial);
        capacity_churn(trial);
    }
}
