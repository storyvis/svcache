use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use svcache::{CacheKey, EvictReason, ManualClock, SvCache, TtlMode};

thread_local!(static CLONES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) });

fn clones() -> usize {
    CLONES.with(|c| c.get())
}

#[derive(Debug, PartialEq)]
struct Item {
    id: u64,
    slug: Option<String>,
    n: u64,
}

impl Clone for Item {
    fn clone(&self) -> Self {
        CLONES.with(|c| c.set(c.get() + 1));
        Item {
            id: self.id,
            slug: self.slug.clone(),
            n: self.n,
        }
    }
}

impl CacheKey for Item {
    type Id = u64;
    fn id(&self) -> u64 {
        self.id
    }
    fn slug(&self) -> Option<&str> {
        self.slug.as_deref()
    }
}

fn item(id: u64) -> Item {
    Item {
        id,
        slug: None,
        n: 0,
    }
}

fn slugged(id: u64, slug: &str) -> Item {
    Item {
        id,
        slug: Some(slug.to_string()),
        n: 0,
    }
}

type Log = Arc<Mutex<Vec<(u64, EvictReason)>>>;

fn logging(b: svcache::SvCacheBuilder<Item>) -> (SvCache<Item>, Log) {
    let log: Log = Arc::default();
    let l = log.clone();
    let cache = b
        .on_evict(move |id, _, r| l.lock().unwrap().push((*id, r)))
        .build();
    (cache, log)
}

fn take(log: &Log) -> Vec<(u64, EvictReason)> {
    std::mem::take(&mut *log.lock().unwrap())
}

#[test]
fn eviction_without_hits_is_fifo() {
    let (c, log) = logging(SvCache::builder().max_entries(3));
    for i in 1..=6 {
        c.insert(item(i));
    }
    assert_eq!(c.len(), 3);
    let cap = EvictReason::Capacity;
    assert_eq!(take(&log), vec![(1, cap), (2, cap), (3, cap)]);
}

#[test]
fn sieve_keeps_visited_entries() {
    let (c, log) = logging(SvCache::builder().max_entries(3));
    for i in 1..=3 {
        c.insert(item(i));
    }
    assert!(c.get_by_id(1).is_some());
    c.insert(item(4)); // 1 visited -> bit cleared, 2 evicted, hand at 3
    c.insert(item(5)); // 3 evicted, hand at 4
    c.insert(item(6)); // 4 (new, unvisited) evicted; 1 survives
    let cap = EvictReason::Capacity;
    assert_eq!(take(&log), vec![(2, cap), (3, cap), (4, cap)]);
    for i in [1, 5, 6] {
        assert!(c.get_with(&i, |_| ()).is_some());
    }
}

#[test]
fn sieve_all_visited_evicts_oldest() {
    let (c, log) = logging(SvCache::builder().max_entries(3));
    for i in 1..=3 {
        c.insert(item(i));
        c.get_with(&i, |_| ());
    }
    c.insert(item(4));
    assert_eq!(take(&log), vec![(1, EvictReason::Capacity)]);
}

#[test]
fn eviction_prefers_expired_over_scan() {
    let clock = ManualClock::new();
    let (c, log) = logging(
        SvCache::builder()
            .ttl(Duration::from_secs(10))
            .max_entries(2)
            .cleanup_budget(0)
            .clock(clock.clone()),
    );
    c.insert(item(1));
    clock.advance(Duration::from_secs(11));
    c.insert(item(2));
    c.insert(item(3));
    assert_eq!(take(&log), vec![(1, EvictReason::Expired)]);
    assert_eq!(c.metadata().evicted_count, 1);
}

#[test]
fn limit_is_exact_under_concurrent_inserts() {
    let c = Arc::new(SvCache::<Item>::with_limit(100));
    let hs: Vec<_> = (0..8u64)
        .map(|t| {
            let c = c.clone();
            std::thread::spawn(move || {
                for i in 0..2_000 {
                    c.insert(item(t * 1_000_000 + i));
                    assert!(c.len() <= 100);
                }
            })
        })
        .collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    assert_eq!(c.len(), 100);
}

#[test]
fn fixed_ttl_with_manual_clock() {
    let clock = ManualClock::new();
    let (c, log) = logging(
        SvCache::builder()
            .ttl(Duration::from_secs(10))
            .clock(clock.clone()),
    );
    c.insert(slugged(1, "a"));
    clock.advance(Duration::from_secs(10));
    assert!(c.get_by_id(1).is_some(), "alive at exactly ttl");
    clock.advance(Duration::from_millis(1));
    assert!(c.get_by_slug("a").is_none());
    assert!(c.get_by_id(1).is_none());
    assert_eq!(take(&log), vec![(1, EvictReason::Expired)]);
    assert_eq!(c.len(), 0);
}

#[test]
fn reinsert_resets_fixed_expiry() {
    let clock = ManualClock::new();
    let c = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .clock(clock.clone())
        .build();
    c.insert(item(1));
    clock.advance(Duration::from_secs(8));
    c.insert(item(1));
    clock.advance(Duration::from_secs(8));
    assert!(c.get_by_id(1).is_some());
}

#[test]
fn sliding_ttl_refreshes_on_access() {
    let clock = ManualClock::new();
    let c = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .ttl_mode(TtlMode::Sliding)
        .clock(clock.clone())
        .build();
    c.insert(item(1));
    c.insert(item(2));
    for _ in 0..3 {
        clock.advance(Duration::from_secs(8));
        assert!(c.get_with(&1, |_| ()).is_some());
        assert_eq!(c.update(&2, |v| v.n += 1), Some(()));
    }
    clock.advance(Duration::from_secs(11));
    assert!(c.get_with(&1, |_| ()).is_none());
    assert!(c.get_with(&2, |_| ()).is_none());
}

#[test]
fn fixed_ttl_does_not_slide_but_touch_does() {
    let clock = ManualClock::new();
    let c = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .clock(clock.clone())
        .build();
    c.insert(item(1));
    c.insert(item(2));
    clock.advance(Duration::from_secs(8));
    assert!(c.get_by_id(1).is_some());
    assert!(c.touch(&2));
    clock.advance(Duration::from_secs(8));
    assert!(c.get_by_id(1).is_none());
    assert!(c.get_by_id(2).is_some());
    assert!(!c.touch(&1));
    assert!(!c.touch(&99));
}

#[test]
fn get_with_update_and_get_or_insert_do_not_clone() {
    let c = SvCache::new();
    c.insert(slugged(1, "a"));
    let before = clones();
    assert_eq!(c.update(&1, |v| v.n += 5), Some(()));
    assert_eq!(c.get_with(&1, |v| v.n), Some(5));
    assert_eq!(c.get_or_insert_with(1, || item(1), |v| v.n), 5);
    c.get_or_insert_with(2, || item(2), |v| v.n += 1);
    assert_eq!(c.get_with(&2, |v| v.n), Some(1));
    assert_eq!(c.update(&3, |v| v.n), None);
    assert_eq!(clones(), before);
}

#[test]
fn get_or_insert_with_loses_no_updates() {
    let c = Arc::new(SvCache::<Item>::new());
    let makes = Arc::new(AtomicUsize::new(0));
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let (c, makes) = (c.clone(), makes.clone());
            std::thread::spawn(move || {
                for _ in 0..1_000 {
                    c.get_or_insert_with(
                        7,
                        || {
                            makes.fetch_add(1, Ordering::Relaxed);
                            item(7)
                        },
                        |v| v.n += 1,
                    );
                }
            })
        })
        .collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    assert_eq!(c.get_with(&7, |v| v.n), Some(8_000));
    assert_eq!(makes.load(Ordering::Relaxed), 1);
}

#[test]
fn get_or_insert_with_replaces_expired() {
    let clock = ManualClock::new();
    let (c, log) = logging(
        SvCache::builder()
            .ttl(Duration::from_secs(1))
            .clock(clock.clone()),
    );
    c.get_or_insert_with(1, || item(1), |v| v.n = 10);
    clock.advance(Duration::from_secs(2));
    assert_eq!(c.get_or_insert_with(1, || item(1), |v| v.n), 0);
    assert_eq!(take(&log), vec![(1, EvictReason::Expired)]);
}

#[test]
fn remove_drops_slug_mapping() {
    let (c, log) = logging(SvCache::builder());
    c.insert(slugged(1, "a"));
    assert_eq!(c.remove(&1).map(|v| v.id), Some(1));
    assert!(c.remove(&1).is_none());
    assert!(c.get_by_slug("a").is_none());
    assert!(c.is_empty());
    c.insert(slugged(2, "a"));
    assert_eq!(c.get_by_slug("a").map(|v| v.id), Some(2));
    assert_eq!(take(&log), vec![(1, EvictReason::Removed)]);
}

#[test]
fn slug_change_on_reinsert_updates_index() {
    let c = SvCache::new();
    c.insert(slugged(1, "old"));
    c.insert(slugged(1, "new"));
    assert!(c.get_by_slug("old").is_none());
    assert_eq!(c.get_by_slug("new").map(|v| v.id), Some(1));
}

#[test]
fn slug_owned_by_later_id_survives_removal_of_earlier() {
    let c = SvCache::new();
    c.insert(slugged(1, "s"));
    c.insert(slugged(2, "s"));
    c.remove(&1);
    assert_eq!(c.get_by_slug("s").map(|v| v.id), Some(2));
}

#[test]
fn slug_cleanup_on_eviction() {
    let c = SvCache::with_limit(2);
    c.insert(slugged(1, "a"));
    c.insert(slugged(2, "b"));
    c.insert(slugged(3, "c"));
    assert!(c.get_by_slug("a").is_none());
    c.insert(slugged(4, "a"));
    assert_eq!(c.get_by_slug("a").map(|v| v.id), Some(4));
}

#[test]
fn slug_lookups_never_return_mismatched_entry() {
    let c = Arc::new(SvCache::<Item>::with_limit(64));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writers: Vec<_> = (0..4u64)
        .map(|t| {
            let c = c.clone();
            std::thread::spawn(move || {
                for i in 0..20_000u64 {
                    let id = t * 1_000_000 + i;
                    c.insert(slugged(id, &format!("s{}", i % 97)));
                    if i % 3 == 0 {
                        c.remove(&(id - id.min(5)));
                    }
                }
            })
        })
        .collect();
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let (c, stop) = (c.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut hits = 0;
                while !stop.load(Ordering::Relaxed) {
                    for s in 0..97 {
                        let slug = format!("s{s}");
                        if let Some(v) = c.get_by_slug(&slug) {
                            assert_eq!(v.slug.as_deref(), Some(slug.as_str()));
                            hits += 1;
                        }
                    }
                }
                hits
            })
        })
        .collect();
    writers.into_iter().for_each(|h| h.join().unwrap());
    stop.store(true, Ordering::Relaxed);
    let hits: usize = readers.into_iter().map(|h| h.join().unwrap()).sum();
    assert!(hits > 0);
    assert!(c.len() <= 64);
}

#[test]
fn callback_reasons_for_clear_and_load() {
    let (c, log) = logging(SvCache::builder());
    c.insert(item(1));
    c.insert(item(2));
    c.clear();
    let mut got = take(&log);
    got.sort_by_key(|e| e.0);
    assert_eq!(
        got,
        vec![(1, EvictReason::Cleared), (2, EvictReason::Cleared)]
    );
    c.load(vec![item(3), item(4)]);
    assert!(take(&log).is_empty());
    c.load(vec![item(4), item(5)]);
    assert_eq!(take(&log), vec![(3, EvictReason::Cleared)]);
    assert_eq!(c.len(), 2);
}

#[test]
fn callback_may_reenter_cache() {
    let c: Arc<Mutex<Option<Arc<SvCache<Item>>>>> = Arc::default();
    let c2 = c.clone();
    let cache = Arc::new(
        SvCache::builder()
            .max_entries(1)
            .on_evict(move |id, _, _| {
                if let Some(c) = c2.lock().unwrap().as_ref() {
                    assert!(c.get_by_id(*id).is_none());
                    let _ = c.len();
                }
            })
            .build(),
    );
    *c.lock().unwrap() = Some(cache.clone());
    cache.insert(item(1));
    cache.insert(item(2));
    assert_eq!(cache.len(), 1);
    *c.lock().unwrap() = None;
}

#[test]
fn budgeted_cleanup() {
    let clock = ManualClock::new();
    let (c, log) = logging(
        SvCache::builder()
            .ttl(Duration::from_secs(1))
            .cleanup_budget(0)
            .clock(clock.clone()),
    );
    for i in 0..100 {
        c.insert(item(i));
    }
    clock.advance(Duration::from_secs(2));
    c.insert(item(1000));
    assert_eq!(c.evict_expired_budget(10), 10);
    assert_eq!(c.len(), 91);
    let mut total = 10;
    while total < 100 {
        let n = c.evict_expired_budget(10);
        assert!(n <= 10);
        total += n;
    }
    assert_eq!(total, 100);
    assert_eq!(c.len(), 1);
    assert!(c.get_by_id(1000).is_some());
    assert_eq!(take(&log).len(), 100);
    assert_eq!(c.evict_expired(), 0);
}

#[test]
fn inline_cleanup_is_time_gated_and_bounded() {
    let clock = ManualClock::new();
    let c = SvCache::builder()
        .ttl(Duration::from_secs(10))
        .cleanup_interval(Duration::from_secs(1))
        .cleanup_budget(5)
        .clock(clock.clone())
        .build();
    for i in 0..50 {
        c.insert(item(i));
    }
    c.get_by_id(0);
    assert_eq!(c.len(), 50);
    clock.advance(Duration::from_secs(11));
    c.insert(item(100)); // sweeps 5 slots
    assert_eq!(c.len(), 46);
    c.get_by_id(100); // same instant: no sweep
    assert_eq!(c.len(), 46);
    clock.advance(Duration::from_secs(1));
    c.get_by_id(100);
    assert_eq!(c.len(), 41);
    assert_eq!(c.evict_expired(), 40);
    assert_eq!(c.len(), 1);
}

#[test]
fn load_has_no_empty_window() {
    let c = Arc::new(SvCache::<Item>::new());
    c.load((0..500).map(item).collect());
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reader = {
        let (c, stop) = (c.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                for i in 0..500 {
                    assert!(c.get_with(&i, |_| ()).is_some(), "miss on {i} during load");
                }
            }
        })
    };
    for round in 0..50 {
        c.load(
            (0..500)
                .map(|i| Item {
                    n: round,
                    ..item(i)
                })
                .collect(),
        );
    }
    stop.store(true, Ordering::Relaxed);
    reader.join().unwrap();
    assert_eq!(c.get_with(&0, |v| v.n), Some(49));
}

#[test]
fn load_enforces_limit() {
    let c = SvCache::<Item>::with_limit(3);
    c.load((0..10).map(item).collect());
    assert_eq!(c.len(), 3);
}

#[cfg(feature = "stats")]
#[test]
fn striped_stats_sum_across_threads() {
    let c = Arc::new(SvCache::<Item>::new());
    c.insert(item(1));
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let c = c.clone();
            std::thread::spawn(move || {
                for _ in 0..1_000 {
                    c.get_with(&1, |_| ());
                    c.get_with(&2, |_| ());
                }
            })
        })
        .collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    let m = c.metadata();
    assert_eq!((m.hit_count, m.miss_count), (8_000, 8_000));
}
