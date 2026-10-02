//! svcache - Isomorphic Dual-Index Cache for WASM and Native Runtimes
//!
//! Automatically toggles between a sharded `DashMap` architecture on native
//! multi-threaded targets (like Tokio) and a lean `RwLock<HashMap>` structure on
//! single-threaded WASM environments (like Cloudflare Workers).
//!
//! # Semantics
//!
//! * **Eviction**: [SIEVE](https://cachemon.github.io/SIEVE-website/). Every entry carries a
//!   `visited` bit set on access; when the cache is at capacity a hand walks from the oldest
//!   entry towards the newest, clearing set bits and evicting the first entry whose bit is
//!   clear (an expired entry is evicted regardless of its bit). O(1) amortized. Without
//!   hits the policy is exact FIFO.
//! * **Clock**: monotonic (`std::time::Instant` natively, `performance.now()` via `web-time`
//!   on wasm32), millisecond resolution. An entry expires once more than `ttl` has elapsed
//!   since it was inserted (or, with [`TtlMode::Sliding`], last accessed).
//! * **Cleanup**: expired entries are invisible to lookups immediately. They are reclaimed
//!   when a lookup hits them, by the eviction hand, by a bounded inline sweep that runs at
//!   most once per [`SvCacheBuilder::cleanup_interval`], and by
//!   [`SvCache::evict_expired_budget`] / [`SvCache::evict_expired`].
//! * **Concurrency**: lookups (`get_*`, `update`, `touch`) take only the shard lock of the
//!   entry they touch. Structural changes (insert of a new id, remove, eviction, sweep,
//!   `load`, `clear`) serialize on one mutex that guards the eviction order.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashSet;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Trait for types that can be cached with an ID and optional slug lookups.
pub trait CacheKey: Clone + Send + Sync + 'static {
    /// The primary identifier type (e.g., u64, String, UUID).
    type Id: Hash + Eq + Clone + Send + Sync + 'static;

    /// Get the unique primary key ID of this item.
    fn id(&self) -> Self::Id;

    /// Get the optional secondary slug/name for auxiliary lookup.
    fn slug(&self) -> Option<&str> {
        None
    }
}

/// Metadata about the cache state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub initialized: bool,
    pub loaded_at: Option<chrono::DateTime<Utc>>,
    pub count: usize,
    pub ttl_secs: Option<u64>,
    pub max_entries: Option<usize>,
    pub evicted_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_rate: f64,
}

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            initialized: false,
            loaded_at: None,
            count: 0,
            ttl_secs: None,
            max_entries: None,
            evicted_count: 0,
            hit_count: 0,
            miss_count: 0,
            hit_rate: 0.0,
        }
    }
}

/// Why an entry left the cache, passed to the [`SvCacheBuilder::on_evict`] callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictReason {
    /// Its TTL elapsed.
    Expired,
    /// Evicted by SIEVE to make room for a new id.
    Capacity,
    /// Removed through [`SvCache::remove`].
    Removed,
    /// Dropped by [`SvCache::clear`], or not present in a [`SvCache::load`].
    Cleared,
}

/// How an entry's expiry is computed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TtlMode {
    /// Expiry is set on insert (or re-insert) only.
    #[default]
    Fixed,
    /// Every successful lookup or update also pushes expiry to `now + ttl`.
    Sliding,
}

/// A manually advanced clock for deterministic tests, or for hosts that drive time
/// themselves. Clones share the same time.
#[derive(Debug, Clone, Default)]
pub struct ManualClock(Arc<AtomicU64>);

impl ManualClock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn advance(&self, by: Duration) {
        self.0.fetch_add(duration_ms(by), Relaxed);
    }

    pub fn now(&self) -> Duration {
        Duration::from_millis(self.0.load(Relaxed))
    }
}

enum Clock {
    Monotonic(Instant),
    Manual(ManualClock),
}

impl Clock {
    #[inline]
    fn now_ms(&self) -> u64 {
        match self {
            Clock::Monotonic(epoch) => epoch.elapsed().as_millis() as u64,
            Clock::Manual(c) => c.0.load(Relaxed),
        }
    }
}

fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_nanos().div_ceil(1_000_000)).unwrap_or(u64::MAX)
}

#[cfg(feature = "stats")]
mod stats {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

    const STRIPES: usize = 16;

    #[repr(align(128))]
    #[derive(Default)]
    struct Line(AtomicU64);

    /// Striped counter: concurrent threads increment different cache lines.
    #[derive(Default)]
    pub struct Counter([Line; STRIPES]);

    #[inline]
    fn stripe() -> usize {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        thread_local!(static IDX: usize = NEXT.fetch_add(1, Relaxed) % STRIPES);
        IDX.with(|i| *i)
    }

    impl Counter {
        #[inline]
        pub fn incr(&self) {
            self.0[stripe()].0.fetch_add(1, Relaxed);
        }

        pub fn sum(&self) -> u64 {
            self.0.iter().map(|l| l.0.load(Relaxed)).sum()
        }

        pub fn reset(&self) {
            self.0.iter().for_each(|l| l.0.store(0, Relaxed));
        }
    }
}

#[cfg(not(feature = "stats"))]
mod stats {
    #[derive(Default)]
    pub struct Counter(());

    impl Counter {
        #[inline]
        pub fn incr(&self) {}

        pub fn sum(&self) -> u64 {
            0
        }

        pub fn reset(&self) {}
    }
}

// =========================================================================
// MAP ENGINE: sharded DashMap natively, RwLock<HashMap> on wasm32
// =========================================================================
#[cfg(not(target_arch = "wasm32"))]
mod map {
    use super::*;
    use dashmap::DashMap;

    pub struct Map<K, V>(DashMap<K, V>);

    impl<K: Hash + Eq, V> Map<K, V> {
        pub fn new() -> Self {
            Self(DashMap::new())
        }

        #[inline]
        pub fn read<Q, R>(&self, k: &Q, f: impl FnOnce(&V) -> R) -> Option<R>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            self.0.get(k).map(|r| f(r.value()))
        }

        #[inline]
        pub fn write<R>(&self, k: &K, f: impl FnOnce(&mut V) -> R) -> Option<R> {
            self.0.get_mut(k).map(|mut r| f(r.value_mut()))
        }

        pub fn insert(&self, k: K, v: V) -> Option<V> {
            self.0.insert(k, v)
        }

        pub fn remove_if<Q>(&self, k: &Q, f: impl FnOnce(&V) -> bool) -> Option<(K, V)>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            self.0.remove_if(k, |_, v| f(v))
        }

        pub fn clear(&self) {
            self.0.clear()
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod map {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{PoisonError, RwLock};

    pub struct Map<K, V>(RwLock<HashMap<K, V>>);

    impl<K: Hash + Eq, V> Map<K, V> {
        pub fn new() -> Self {
            Self(RwLock::new(HashMap::new()))
        }

        #[inline]
        pub fn read<Q, R>(&self, k: &Q, f: impl FnOnce(&V) -> R) -> Option<R>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            self.0
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .get(k)
                .map(f)
        }

        #[inline]
        pub fn write<R>(&self, k: &K, f: impl FnOnce(&mut V) -> R) -> Option<R> {
            self.0
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .get_mut(k)
                .map(f)
        }

        pub fn insert(&self, k: K, v: V) -> Option<V> {
            self.0
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(k, v)
        }

        pub fn remove_if<Q>(&self, k: &Q, f: impl FnOnce(&V) -> bool) -> Option<(K, V)>
        where
            K: Borrow<Q>,
            Q: Hash + Eq + ?Sized,
        {
            let mut m = self.0.write().unwrap_or_else(PoisonError::into_inner);
            if m.get(k).is_some_and(f) {
                m.remove_entry(k)
            } else {
                None
            }
        }

        pub fn clear(&self) {
            self.0
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .clear()
        }
    }
}

use map::Map;

const NEVER: u64 = u64::MAX;
const NIL: usize = usize::MAX;
const DEFAULT_CLEANUP_BUDGET: usize = 64;

struct Entry<T> {
    value: T,
    /// Slug this entry is indexed under; slug lookups only hit when it matches.
    slug: Option<Box<str>>,
    /// Slot in [`Order`]; stable while the entry is in the map.
    node: usize,
    visited: AtomicBool,
    expires_at: AtomicU64,
}

impl<T> Entry<T> {
    #[inline]
    fn expired(&self, now: u64) -> bool {
        now > self.expires_at.load(Relaxed)
    }
}

struct Node<K> {
    key: K,
    newer: usize,
    older: usize,
}

/// Insertion-ordered list (slab-backed, doubly linked) plus the SIEVE hand and sweep cursor.
struct Order<K> {
    slots: Vec<Option<Node<K>>>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
    hand: usize,
    cursor: usize,
    len: usize,
}

impl<K> Order<K> {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            hand: NIL,
            cursor: 0,
            len: 0,
        }
    }

    fn node(&self, idx: usize) -> &Node<K> {
        self.slots[idx].as_ref().expect("live order slot")
    }

    fn node_mut(&mut self, idx: usize) -> &mut Node<K> {
        self.slots[idx].as_mut().expect("live order slot")
    }

    fn push_head(&mut self, key: K) -> usize {
        let node = Node {
            key,
            newer: NIL,
            older: self.head,
        };
        let idx = match self.free.pop() {
            Some(i) => {
                self.slots[i] = Some(node);
                i
            }
            None => {
                self.slots.push(Some(node));
                self.slots.len() - 1
            }
        };
        if self.head != NIL {
            let head = self.head;
            self.node_mut(head).newer = idx;
        }
        self.head = idx;
        if self.tail == NIL {
            self.tail = idx;
        }
        self.len += 1;
        idx
    }

    fn unlink(&mut self, idx: usize) {
        let node = self.slots[idx].take().expect("live order slot");
        match node.newer {
            NIL => self.head = node.older,
            n => self.node_mut(n).older = node.older,
        }
        match node.older {
            NIL => self.tail = node.newer,
            o => self.node_mut(o).newer = node.newer,
        }
        if self.hand == idx {
            self.hand = node.newer;
        }
        self.len -= 1;
        if self.len == 0 {
            *self = Self::new();
        } else {
            self.free.push(idx);
        }
    }
}

type EvictFn<T> = Box<dyn Fn(&<T as CacheKey>::Id, &T, EvictReason) + Send + Sync>;
/// Entries removed under the structural lock, reported once it is released. Values are kept
/// only when there is a callback; otherwise they are dropped on the spot.
struct Evicted<T: CacheKey> {
    items: Vec<(T::Id, T, EvictReason)>,
    keep: bool,
    removed: usize,
    evicted: u64,
}

impl<T: CacheKey> Evicted<T> {
    fn push(&mut self, k: T::Id, v: T, reason: EvictReason) {
        self.removed += 1;
        if matches!(reason, EvictReason::Expired | EvictReason::Capacity) {
            self.evicted += 1;
        }
        if self.keep {
            self.items.push((k, v, reason));
        }
    }
}

// =========================================================================
// BUILDER
// =========================================================================

/// Builder for [`SvCache`], exposing the opt-in behaviours.
pub struct SvCacheBuilder<T: CacheKey> {
    ttl: Option<Duration>,
    max_entries: Option<usize>,
    ttl_mode: TtlMode,
    cleanup_interval: Option<Duration>,
    cleanup_budget: usize,
    clock: Option<ManualClock>,
    on_evict: Option<EvictFn<T>>,
}

impl<T: CacheKey> SvCacheBuilder<T> {
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = Some(max);
        self
    }

    pub fn ttl_mode(mut self, mode: TtlMode) -> Self {
        self.ttl_mode = mode;
        self
    }

    /// Minimum time between inline sweeps. Default `min(ttl / 4, 1s)`, at least 1 ms.
    pub fn cleanup_interval(mut self, interval: Duration) -> Self {
        self.cleanup_interval = Some(interval);
        self
    }

    /// Entries examined by one inline sweep. Default 64; 0 disables inline sweeps
    /// (call [`SvCache::evict_expired_budget`] off the hot path instead).
    pub fn cleanup_budget(mut self, entries: usize) -> Self {
        self.cleanup_budget = entries;
        self
    }

    /// Use a manually advanced clock instead of the monotonic system clock.
    pub fn clock(mut self, clock: ManualClock) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Called once for every entry that leaves the cache other than by being overwritten.
    ///
    /// Runs after the entry is gone from both indices, with no cache lock held, on the
    /// thread that caused the removal, so it may call back into the cache. Callbacks from
    /// different threads are not ordered with respect to each other.
    pub fn on_evict(mut self, f: impl Fn(&T::Id, &T, EvictReason) + Send + Sync + 'static) -> Self {
        self.on_evict = Some(Box::new(f));
        self
    }

    pub fn build(self) -> SvCache<T> {
        let interval = self
            .cleanup_interval
            .or(self.ttl.map(|t| (t / 4).min(Duration::from_secs(1))))
            .unwrap_or(Duration::from_secs(1));
        let interval_ms = duration_ms(interval).max(1);
        let metadata = CacheMetadata {
            ttl_secs: self.ttl.map(|d| d.as_secs()),
            max_entries: self.max_entries,
            ..Default::default()
        };
        let clock = match self.clock {
            Some(c) => Clock::Manual(c),
            None => Clock::Monotonic(Instant::now()),
        };
        let next_sweep = clock.now_ms().saturating_add(interval_ms);
        SvCache {
            by_id: Map::new(),
            by_slug: Map::new(),
            order: Mutex::new(Order::new()),
            clock,
            ttl_ms: self.ttl.map(duration_ms),
            ttl_mode: self.ttl_mode,
            max_entries: self.max_entries,
            cleanup_interval_ms: interval_ms,
            cleanup_budget: self.cleanup_budget,
            next_sweep: AtomicU64::new(next_sweep),
            on_evict: self.on_evict,
            metadata: std::sync::RwLock::new(metadata),
            live: AtomicUsize::new(0),
            evicted: AtomicU64::new(0),
            hits: stats::Counter::default(),
            misses: stats::Counter::default(),
        }
    }
}

// =========================================================================
// UNIFIED FRONT-FACING ARCHITECTURE
// =========================================================================
pub struct SvCache<T: CacheKey> {
    by_id: Map<T::Id, Entry<T>>,
    by_slug: Map<Box<str>, T::Id>,
    // Lock order: `order`, then at most one map shard at a time.
    order: Mutex<Order<T::Id>>,
    clock: Clock,
    ttl_ms: Option<u64>,
    ttl_mode: TtlMode,
    max_entries: Option<usize>,
    cleanup_interval_ms: u64,
    cleanup_budget: usize,
    next_sweep: AtomicU64,
    on_evict: Option<EvictFn<T>>,
    metadata: std::sync::RwLock<CacheMetadata>,
    live: AtomicUsize,
    evicted: AtomicU64,
    hits: stats::Counter,
    misses: stats::Counter,
}

impl<T: CacheKey> SvCache<T> {
    /// Creates an entirely unbounded unified cache context.
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn with_limit(max_entries: usize) -> Self {
        Self::builder().max_entries(max_entries).build()
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self::builder().ttl(ttl).build()
    }

    pub fn with_ttl_and_limit(ttl: Duration, max_entries: usize) -> Self {
        Self::builder().ttl(ttl).max_entries(max_entries).build()
    }

    pub fn builder() -> SvCacheBuilder<T> {
        SvCacheBuilder {
            ttl: None,
            max_entries: None,
            ttl_mode: TtlMode::Fixed,
            cleanup_interval: None,
            cleanup_budget: DEFAULT_CLEANUP_BUDGET,
            clock: None,
            on_evict: None,
        }
    }

    /// Inserts or replaces the item. Replacing resets its expiry and counts as an access.
    pub fn insert(&self, item: T) {
        self.insert_iter(std::iter::once(item));
    }

    pub fn insert_many(&self, items: Vec<T>) {
        self.insert_iter(items);
    }

    fn insert_iter(&self, items: impl IntoIterator<Item = T>) {
        let now = self.now();
        self.maybe_sweep(now);
        let mut out = self.evicted_sink();
        let mut st = self.lock();
        for item in items {
            self.insert_locked(&mut st, item, now, true, &mut out);
        }
        self.unlock(st);
        self.notify(out);
    }

    pub fn get_by_id(&self, id: T::Id) -> Option<T> {
        self.get_with(&id, Clone::clone)
    }

    pub fn get_by_slug(&self, slug: &str) -> Option<T> {
        let now = self.now();
        self.maybe_sweep(now);
        let Some(id) = self.by_slug.read(slug, Clone::clone) else {
            self.misses.incr();
            return None;
        };
        let r = self
            .by_id
            .read(&id, |e| {
                if e.slug.as_deref() != Some(slug) {
                    return None;
                }
                Some(self.hit(e, now).then(|| e.value.clone()).ok_or(()))
            })
            .flatten();
        self.finish(&id, r, now)
    }

    /// Runs `f` on the cached value under the entry's shard read lock, without cloning it.
    ///
    /// `f` must not access this cache (it could deadlock on the same shard).
    #[inline]
    pub fn get_with<R>(&self, id: &T::Id, f: impl FnOnce(&T) -> R) -> Option<R> {
        let now = self.now();
        self.maybe_sweep(now);
        let r = self
            .by_id
            .read(id, |e| self.hit(e, now).then(|| f(&e.value)).ok_or(()));
        self.finish(id, r, now)
    }

    /// Mutates the cached value in place under the entry's shard write lock.
    ///
    /// `f` must not change the item's id or slug (the indices keep the values from insert;
    /// re-`insert` to change them) and must not access this cache.
    pub fn update<R>(&self, id: &T::Id, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let now = self.now();
        self.maybe_sweep(now);
        let r = self
            .by_id
            .write(id, |e| self.hit(e, now).then(|| f(&mut e.value)).ok_or(()));
        self.finish(id, r, now)
    }

    /// Runs `f` on the live entry for `id`, inserting `make()` first if there is none.
    ///
    /// Concurrent callers for the same id share one `make()` result and their `f` calls are
    /// serialized, so no update is lost. `make` runs under the structural lock (keep it
    /// cheap) and must return an item whose id is `id`. Same restrictions on `f` as
    /// [`SvCache::update`].
    pub fn get_or_insert_with<R>(
        &self,
        id: T::Id,
        make: impl FnOnce() -> T,
        f: impl FnOnce(&mut T) -> R,
    ) -> R {
        let now = self.now();
        self.maybe_sweep(now);
        let mut f = Some(f);
        let mut existing = |e: &mut Entry<T>| {
            self.hit(e, now)
                .then(|| (f.take().expect("f unused"))(&mut e.value))
        };
        if let Some(Some(r)) = self.by_id.write(&id, &mut existing) {
            self.hits.incr();
            return r;
        }
        let mut out = self.evicted_sink();
        let mut st = self.lock();
        let r = match self.by_id.write(&id, &mut existing) {
            Some(Some(r)) => {
                self.hits.incr();
                r
            }
            _ => {
                if let Some((k, e)) = self.remove_locked(&mut st, &id, |e| e.expired(now)) {
                    out.push(k, e.value, EvictReason::Expired);
                }
                self.misses.incr();
                let mut item = make();
                debug_assert!(item.id() == id, "make() returned an item with another id");
                let r = (f.take().expect("f unused"))(&mut item);
                self.insert_locked(&mut st, item, now, true, &mut out);
                r
            }
        };
        self.unlock(st);
        self.notify(out);
        r
    }

    /// Marks the entry accessed and pushes its expiry to `now + ttl` (in either TTL mode).
    /// Returns false if the id is absent or expired.
    pub fn touch(&self, id: &T::Id) -> bool {
        let now = self.now();
        let exp = self.expiry(now);
        let r = self.by_id.read(id, |e| {
            if e.expired(now) {
                return false;
            }
            e.visited.store(true, Relaxed);
            e.expires_at.store(exp, Relaxed);
            true
        });
        if r == Some(false) {
            self.reap(id, now);
        }
        r == Some(true)
    }

    /// Removes the entry and its slug mapping. Returns `None` if absent or already expired
    /// (an expired entry is still removed, with reason `Expired`).
    pub fn remove(&self, id: &T::Id) -> Option<T> {
        let now = self.now();
        let mut st = self.lock();
        let removed = self.remove_locked(&mut st, id, |_| true);
        self.unlock(st);
        let (k, e) = removed?;
        if e.expired(now) {
            let mut out = self.evicted_sink();
            out.push(k, e.value, EvictReason::Expired);
            self.notify(out);
            return None;
        }
        if let Some(cb) = &self.on_evict {
            cb(&k, &e.value, EvictReason::Removed);
        }
        Some(e.value)
    }

    /// Removes every expired entry (full scan under the structural lock).
    pub fn evict_expired(&self) -> usize {
        self.evict_expired_budget(usize::MAX)
    }

    /// Examines at most `max_entries` entries, continuing where the previous sweep stopped,
    /// and removes the expired ones. Returns the number removed.
    pub fn evict_expired_budget(&self, max_entries: usize) -> usize {
        if self.ttl_ms.is_none() {
            return 0;
        }
        let now = self.now();
        let mut out = self.evicted_sink();
        let mut st = self.lock();
        self.sweep_locked(&mut st, now, max_entries, &mut out);
        self.unlock(st);
        let n = out.removed;
        self.notify(out);
        n
    }

    /// Replaces the contents with `items`.
    ///
    /// Not a snapshot swap: `items` are upserted first, then ids absent from `items` are
    /// removed (reason `Cleared`). An id present before and after never misses while `load`
    /// runs, but a concurrent reader may see old values for some ids and new values for
    /// others. The entry limit is enforced afterwards. Resets hit/miss counters.
    pub fn load(&self, items: Vec<T>) {
        let now = self.now();
        let mut out = self.evicted_sink();
        let mut st = self.lock();
        let mut keep = HashSet::with_capacity(items.len());
        for item in items {
            keep.insert(item.id());
            self.insert_locked(&mut st, item, now, false, &mut out);
        }
        let stale: Vec<T::Id> = st
            .slots
            .iter()
            .flatten()
            .filter(|n| !keep.contains(&n.key))
            .map(|n| n.key.clone())
            .collect();
        drop(keep);
        for k in stale {
            if let Some((k, e)) = self.remove_locked(&mut st, &k, |_| true) {
                out.push(k, e.value, EvictReason::Cleared);
            }
        }
        if let Some(max) = self.max_entries {
            while st.len > max && self.evict_one(&mut st, now, &mut out) {}
        }
        self.unlock(st);
        self.hits.reset();
        self.misses.reset();
        if let Ok(mut meta) = self.metadata.write() {
            meta.initialized = true;
            meta.loaded_at = Some(Utc::now());
        }
        self.notify(out);
    }

    /// Removes all entries (reason `Cleared`) and resets hit/miss counters.
    pub fn clear(&self) {
        let mut out = self.evicted_sink();
        let mut st = self.lock();
        if self.on_evict.is_some() {
            while st.tail != NIL {
                let tail = st.tail;
                let k = st.node(tail).key.clone();
                match self.remove_locked(&mut st, &k, |_| true) {
                    Some((k, e)) => out.push(k, e.value, EvictReason::Cleared),
                    None => st.unlink(tail),
                }
            }
        } else {
            self.by_id.clear();
            self.by_slug.clear();
            *st = Order::new();
        }
        self.unlock(st);
        self.hits.reset();
        self.misses.reset();
        self.notify(out);
    }

    /// Number of entries, including expired ones not yet reclaimed.
    pub fn len(&self) -> usize {
        self.live.load(Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cache statistics. Hit/miss counts stay 0 without the `stats` feature.
    pub fn metadata(&self) -> CacheMetadata {
        let mut meta = self.metadata.read().map(|m| m.clone()).unwrap_or_default();
        meta.count = self.len();
        meta.evicted_count = self.evicted.load(Relaxed);
        meta.hit_count = self.hits.sum();
        meta.miss_count = self.misses.sum();
        let total = meta.hit_count + meta.miss_count;
        meta.hit_rate = if total == 0 {
            0.0
        } else {
            meta.hit_count as f64 / total as f64
        };
        meta
    }

    // ---------------------------------------------------------------------

    #[inline]
    fn now(&self) -> u64 {
        if self.ttl_ms.is_some() {
            self.clock.now_ms()
        } else {
            0
        }
    }

    #[inline]
    fn expiry(&self, now: u64) -> u64 {
        self.ttl_ms.map_or(NEVER, |t| now.saturating_add(t))
    }

    /// Lookup-side bookkeeping for a found entry; false if it is expired.
    #[inline]
    fn hit(&self, e: &Entry<T>, now: u64) -> bool {
        if e.expired(now) {
            return false;
        }
        if !e.visited.load(Relaxed) {
            e.visited.store(true, Relaxed);
        }
        if self.ttl_mode == TtlMode::Sliding {
            // Millisecond resolution keeps hot entries from rewriting this line on every hit.
            let exp = self.expiry(now);
            if e.expires_at.load(Relaxed) != exp {
                e.expires_at.store(exp, Relaxed);
            }
        }
        true
    }

    #[inline]
    fn finish<R>(&self, id: &T::Id, r: Option<Result<R, ()>>, now: u64) -> Option<R> {
        match r {
            Some(Ok(v)) => {
                self.hits.incr();
                Some(v)
            }
            Some(Err(())) => {
                self.misses.incr();
                self.reap(id, now);
                None
            }
            None => {
                self.misses.incr();
                None
            }
        }
    }

    /// Reclaims an expired entry seen by a lookup, unless another thread holds the lock.
    #[cold]
    fn reap(&self, id: &T::Id, now: u64) {
        let Some(mut st) = self.try_lock() else {
            return;
        };
        let removed = self.remove_locked(&mut st, id, |e| e.expired(now));
        self.unlock(st);
        if let Some((k, e)) = removed {
            let mut out = self.evicted_sink();
            out.push(k, e.value, EvictReason::Expired);
            self.notify(out);
        }
    }

    #[inline]
    fn maybe_sweep(&self, now: u64) {
        if self.cleanup_budget != 0 && self.ttl_ms.is_some() && now >= self.next_sweep.load(Relaxed)
        {
            self.sweep_due(now);
        }
    }

    #[cold]
    fn sweep_due(&self, now: u64) {
        let next = self.next_sweep.load(Relaxed);
        let claimed = now >= next
            && self
                .next_sweep
                .compare_exchange(next, now + self.cleanup_interval_ms, Relaxed, Relaxed)
                .is_ok();
        if !claimed {
            return;
        }
        let Some(mut st) = self.try_lock() else {
            return;
        };
        let mut out = self.evicted_sink();
        self.sweep_locked(&mut st, now, self.cleanup_budget, &mut out);
        self.unlock(st);
        self.notify(out);
    }

    fn lock(&self) -> MutexGuard<'_, Order<T::Id>> {
        self.order.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn try_lock(&self) -> Option<MutexGuard<'_, Order<T::Id>>> {
        match self.order.try_lock() {
            Ok(g) => Some(g),
            Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    fn unlock(&self, st: MutexGuard<'_, Order<T::Id>>) {
        self.live.store(st.len, Relaxed);
    }

    fn evicted_sink(&self) -> Evicted<T> {
        Evicted {
            items: Vec::new(),
            keep: self.on_evict.is_some(),
            removed: 0,
            evicted: 0,
        }
    }

    fn notify(&self, out: Evicted<T>) {
        if out.evicted > 0 {
            self.evicted.fetch_add(out.evicted, Relaxed);
        }
        if let Some(cb) = &self.on_evict {
            for (k, v, r) in &out.items {
                cb(k, v, *r);
            }
        }
    }

    fn insert_locked(
        &self,
        st: &mut Order<T::Id>,
        item: T,
        now: u64,
        enforce_limit: bool,
        out: &mut Evicted<T>,
    ) {
        let id = item.id();
        let slug: Option<Box<str>> = item.slug().map(Into::into);
        let exp = self.expiry(now);
        let mut item = Some(item);
        let replaced = self.by_id.write(&id, |e| {
            *e.expires_at.get_mut() = exp;
            *e.visited.get_mut() = true;
            let old = std::mem::replace(&mut e.value, item.take().expect("item"));
            (old, std::mem::replace(&mut e.slug, slug.clone()))
        });
        match replaced {
            Some((_old_value, old_slug)) => {
                if let Some(old) = old_slug
                    && Some(&old) != slug.as_ref()
                {
                    self.by_slug.remove_if(&*old, |v| *v == id);
                }
            }
            None => {
                if enforce_limit && let Some(max) = self.max_entries {
                    while st.len >= max && self.evict_one(st, now, out) {}
                }
                let entry = Entry {
                    value: item.take().expect("item"),
                    slug: slug.clone(),
                    node: st.push_head(id.clone()),
                    visited: AtomicBool::new(false),
                    expires_at: AtomicU64::new(exp),
                };
                self.by_id.insert(id.clone(), entry);
            }
        }
        if let Some(s) = slug {
            self.by_slug.insert(s, id);
        }
    }

    fn remove_locked(
        &self,
        st: &mut Order<T::Id>,
        id: &T::Id,
        pred: impl FnOnce(&Entry<T>) -> bool,
    ) -> Option<(T::Id, Entry<T>)> {
        let (k, e) = self.by_id.remove_if(id, pred)?;
        st.unlink(e.node);
        if let Some(s) = &e.slug {
            self.by_slug.remove_if(&**s, |v| *v == k);
        }
        Some((k, e))
    }

    /// SIEVE: evicts one entry. Returns false only if the cache is empty.
    fn evict_one(&self, st: &mut Order<T::Id>, now: u64, out: &mut Evicted<T>) -> bool {
        // One full lap clears every bit; the bound only matters if concurrent hits keep
        // re-setting them, and then the entry under the hand goes regardless.
        let mut steps = 2 * st.len + 1;
        loop {
            let idx = if st.hand != NIL { st.hand } else { st.tail };
            if idx == NIL {
                return false;
            }
            let node = st.node(idx);
            steps = steps.saturating_sub(1);
            let state = self.by_id.read(&node.key, |e| {
                let expired = e.expired(now);
                let visited = steps > 0 && !expired && e.visited.load(Relaxed);
                if visited {
                    e.visited.store(false, Relaxed);
                }
                (expired, visited)
            });
            match state {
                Some((_, true)) => st.hand = node.newer,
                Some((expired, false)) => {
                    let key = node.key.clone();
                    st.hand = node.newer;
                    if let Some((k, e)) = self.remove_locked(st, &key, |_| true) {
                        let reason = if expired {
                            EvictReason::Expired
                        } else {
                            EvictReason::Capacity
                        };
                        out.push(k, e.value, reason);
                        return true;
                    }
                }
                None => st.unlink(idx),
            }
        }
    }

    fn sweep_locked(&self, st: &mut Order<T::Id>, now: u64, budget: usize, out: &mut Evicted<T>) {
        for _ in 0..budget.min(st.slots.len()) {
            if st.slots.is_empty() {
                break;
            }
            if st.cursor >= st.slots.len() {
                st.cursor = 0;
            }
            let i = st.cursor;
            st.cursor += 1;
            let Some(node) = &st.slots[i] else {
                continue;
            };
            let key = node.key.clone();
            if let Some((k, e)) = self.remove_locked(st, &key, |e| e.expired(now)) {
                out.push(k, e.value, EvictReason::Expired);
            }
        }
    }
}

impl<T: CacheKey> Default for SvCache<T> {
    fn default() -> Self {
        Self::new()
    }
}
