use std::sync::{Arc, Mutex};
use std::time::Duration;
use svcache::{CacheKey, EvictReason, ManualClock, SvCache};

#[derive(Clone, Debug)]
struct Item(u64);

impl CacheKey for Item {
    type Id = u64;
    fn id(&self) -> u64 {
        self.0
    }
}

#[test]
fn cleanup_budget_counts_remaining_entries_after_shrink() {
    let clock = ManualClock::new();
    let cache = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .cleanup_budget(0)
        .clock(clock.clone())
        .build();
    for id in 0..4096 {
        cache.insert(Item(id));
    }
    for id in 0..4095 {
        cache.remove(&id);
    }
    clock.advance(Duration::from_secs(11));
    assert_eq!(cache.evict_expired_budget(1), 1);
    assert!(cache.is_empty());
}

#[test]
fn due_expiry_cleanup_precedes_capacity_eviction() {
    let clock = ManualClock::new();
    let removed = Arc::new(Mutex::new(Vec::new()));
    let report = removed.clone();
    let cache = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .cleanup_interval(Duration::from_secs(11))
        .cleanup_budget(4)
        .max_entries(4)
        .clock(clock.clone())
        .on_evict(move |id, _, why| report.lock().unwrap().push((*id, why)))
        .build();
    for id in 0..3 {
        cache.insert(Item(id));
    }
    clock.advance(Duration::from_secs(8));
    assert!(cache.get_by_id(0).is_some()); // Fixed TTL: visited, still expires at t=10.
    cache.insert(Item(3));
    clock.advance(Duration::from_secs(1));
    cache.insert(Item(4)); // SIEVE clears 0's bit and evicts 1; hand now points at 2.
    clock.advance(Duration::from_secs(1));
    assert!(cache.touch(&2)); // 2 and 3 are live; 0 will expire before the next insert.
    clock.advance(Duration::from_secs(1));
    cache.insert(Item(5));
    assert!(
        cache.get_by_id(3).is_some(),
        "live entry evicted before due expiry sweep"
    );
    assert_eq!(cache.len(), 4);
    assert_eq!(
        *removed.lock().unwrap(),
        vec![(1, EvictReason::Capacity), (0, EvictReason::Expired)]
    );
}

#[test]
fn zero_capacity_never_retains_insert_load_or_computed_entries() {
    let cache = SvCache::with_limit(0);
    cache.insert(Item(1));
    cache.insert(Item(1));
    assert!(cache.is_empty());
    cache.insert_many(vec![Item(2), Item(3)]);
    assert!(cache.is_empty());
    cache.load(vec![Item(4), Item(4), Item(5)]);
    assert!(cache.is_empty());
    assert_eq!(cache.get_or_insert_with(6, || Item(6), |v| v.0), 6);
    assert_eq!(cache.get_or_insert_with(6, || Item(6), |v| v.0 + 1), 7);
    assert!(cache.is_empty());
    assert!(cache.get_by_id(6).is_none());
}

#[test]
fn manual_clock_saturates_instead_of_resurrecting_expired_entries() {
    let clock = ManualClock::new();
    let cache = SvCache::builder()
        .ttl(Duration::from_millis(10))
        .cleanup_budget(0)
        .clock(clock.clone())
        .build();
    cache.insert(Item(1));
    clock.advance(Duration::MAX);
    clock.advance(Duration::from_millis(2));
    assert_eq!(clock.now(), Duration::from_millis(u64::MAX));
    assert!(cache.get_by_id(1).is_none());
}

#[test]
fn cleanup_deadline_does_not_overflow_at_clock_limit() {
    let clock = ManualClock::new();
    let cache = SvCache::builder()
        .ttl(Duration::from_millis(1))
        .cleanup_interval(Duration::MAX)
        .clock(clock.clone())
        .build();
    cache.insert(Item(1));
    clock.advance(Duration::MAX);
    assert!(cache.get_by_id(2).is_none());
    assert!(cache.is_empty());
}

#[test]
fn cursor_remains_live_during_concurrent_removals_and_slot_reuse() {
    let clock = ManualClock::new();
    let cache = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .cleanup_budget(0)
        .clock(clock.clone())
        .build();
    for id in 0..128 {
        cache.insert(Item(id));
    }
    std::thread::scope(|scope| {
        for parity in 0..2 {
            let cache = &cache;
            scope.spawn(move || {
                for id in (parity..128).step_by(2) {
                    cache.remove(&id);
                    if id % 4 == 0 {
                        cache.insert(Item(id + 128));
                    }
                }
            });
        }
        scope.spawn(|| {
            for _ in 0..512 {
                assert_eq!(cache.evict_expired_budget(3), 0);
            }
        });
    });
    assert_eq!(cache.len(), 32);
    clock.advance(Duration::from_secs(11));
    assert_eq!(cache.evict_expired_budget(cache.len()), 32);
    cache.insert(Item(1000));
    cache.clear();
    cache.insert(Item(1001));
    clock.advance(Duration::from_secs(11));
    assert_eq!(cache.evict_expired_budget(1), 1);
}
