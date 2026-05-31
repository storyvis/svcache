//! svcache - Isomorphic Dual-Index Cache for WASM and Native Runtimes
//!
//! Automatically toggles between a lock-free, sharded `DashMap` architecture on native
//! multi-threaded targets (like Tokio) and a lean `RwLock<HashMap>` structure on
//! single-threaded WASM environments (like Cloudflare Workers).

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
    pub loaded_at: Option<DateTime<Utc>>,
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

/// Entry wrapper that includes expiry time.
#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    expires_at: Option<DateTime<Utc>>,
}

impl<T> CacheEntry<T> {
    fn new(value: T, ttl: Option<ChronoDuration>) -> Self {
        Self {
            value,
            expires_at: ttl.map(|d| Utc::now() + d),
        }
    }

    fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Utc::now() > exp)
            .unwrap_or(false)
    }
}

const CLEANUP_PROBABILITY: u64 = 100;

// =========================================================================
// ENGINE CONFIGURATION 1: NATIVE RUNTIMES (Multi-Threaded DashMap Engine)
// =========================================================================
#[cfg(not(target_arch = "wasm32"))]
mod engine {
    use super::*;
    use dashmap::DashMap;

    pub struct CacheEngine<T: CacheKey> {
        pub by_id: DashMap<T::Id, CacheEntry<T>>,
        pub by_slug: DashMap<String, T::Id>,
    }

    impl<T: CacheKey> CacheEngine<T> {
        pub fn new() -> Self {
            Self {
                by_id: DashMap::new(),
                by_slug: DashMap::new(),
            }
        }

        pub fn contains_key(&self, id: &T::Id) -> bool {
            self.by_id.contains_key(id)
        }

        pub fn insert_atomic(&self, item: T, ttl: Option<ChronoDuration>) {
            let id = item.id();
            let entry = CacheEntry::new(item, ttl);

            if let Some(slug) = entry.value.slug() {
                self.by_slug.insert(slug.to_string(), id.clone());
            }
            self.by_id.insert(id, entry);
        }

        pub fn remove_atomic(&self, id: &T::Id) -> Option<T> {
            self.by_id.remove(id).map(|(_, entry)| {
                if let Some(slug) = entry.value.slug() {
                    self.by_slug.remove(slug);
                }
                entry.value
            })
        }

        pub fn remove_by_slug_stale(&self, slug: &str) {
            self.by_slug.remove(slug);
        }

        pub fn clear_all(&self) {
            self.by_id.clear();
            self.by_slug.clear();
        }
    }
}

// =========================================================================
// ENGINE CONFIGURATION 2: WASM RUNTIMES (Single-Threaded RwLock Engine)
// =========================================================================
#[cfg(target_arch = "wasm32")]
mod engine {
    use super::*;
    use std::collections::HashMap;
    use std::sync::RwLock;

    pub struct CacheEngine<T: CacheKey> {
        pub by_id: RwLock<HashMap<T::Id, CacheEntry<T>>>,
        pub by_slug: RwLock<HashMap<String, T::Id>>,
    }

    impl<T: CacheKey> CacheEngine<T> {
        pub fn new() -> Self {
            Self {
                by_id: RwLock::new(HashMap::new()),
                by_slug: RwLock::new(HashMap::new()),
            }
        }

        pub fn contains_key(&self, id: &T::Id) -> bool {
            self.by_id
                .read()
                .map(|m| m.contains_key(id))
                .unwrap_or(false)
        }

        pub fn insert_atomic(&self, item: T, ttl: Option<ChronoDuration>) {
            if let (Ok(mut by_id), Ok(mut by_slug)) = (self.by_id.write(), self.by_slug.write()) {
                let id = item.id();
                let entry = CacheEntry::new(item, ttl);
                if let Some(slug) = entry.value.slug() {
                    by_slug.insert(slug.to_string(), id.clone());
                }
                by_id.insert(id, entry);
            }
        }

        pub fn remove_atomic(&self, id: &T::Id) -> Option<T> {
            if let (Ok(mut by_id), Ok(mut by_slug)) = (self.by_id.write(), self.by_slug.write()) {
                if let Some(entry) = by_id.remove(id) {
                    if let Some(slug) = entry.value.slug() {
                        by_slug.remove(slug);
                    }
                    return Some(entry.value);
                }
            }
            None
        }

        pub fn remove_by_slug_stale(&self, slug: &str) {
            if let Ok(mut by_slug) = self.by_slug.write() {
                by_slug.remove(slug);
            }
        }

        pub fn clear_all(&self) {
            if let Ok(mut by_id) = self.by_id.write() {
                by_id.clear();
            }
            if let Ok(mut by_slug) = self.by_slug.write() {
                by_slug.clear();
            }
        }
    }
}

// =========================================================================
// UNIFIED FRONT-FACING ARCHITECTURE
// =========================================================================
pub struct SvCache<T: CacheKey> {
    inner: engine::CacheEngine<T>,
    metadata: std::sync::RwLock<CacheMetadata>,
    ttl: Option<ChronoDuration>,
    max_entries: Option<usize>,
    access_counter: AtomicU64,
    evicted_counter: AtomicU64,
    hit_counter: AtomicU64,
    miss_counter: AtomicU64,
    live_count: AtomicUsize,
}

impl<T: CacheKey> SvCache<T> {
    /// Creates an entirely unbounded unified cache context.
    pub fn new() -> Self {
        Self::with_options(None, None)
    }

    pub fn with_limit(max_entries: usize) -> Self {
        Self::with_options(None, Some(max_entries))
    }

    pub fn with_ttl(ttl: std::time::Duration) -> Self {
        Self::with_options(ChronoDuration::from_std(ttl).ok(), None)
    }

    pub fn with_ttl_and_limit(ttl: std::time::Duration, max_entries: usize) -> Self {
        Self::with_options(ChronoDuration::from_std(ttl).ok(), Some(max_entries))
    }

    fn with_options(ttl: Option<ChronoDuration>, max_entries: Option<usize>) -> Self {
        let metadata = CacheMetadata {
            ttl_secs: ttl.map(|d| d.num_seconds() as u64),
            max_entries,
            ..Default::default()
        };
        Self {
            inner: engine::CacheEngine::new(),
            metadata: std::sync::RwLock::new(metadata),
            ttl,
            max_entries,
            access_counter: AtomicU64::new(0),
            evicted_counter: AtomicU64::new(0),
            hit_counter: AtomicU64::new(0),
            miss_counter: AtomicU64::new(0),
            live_count: AtomicUsize::new(0),
        }
    }

    pub fn insert(&self, item: T) {
        let id = item.id();
        let is_update = self.inner.contains_key(&id);

        if !is_update {
            if let Some(max) = self.max_entries {
                if self.live_count.load(Ordering::Relaxed) >= max {
                    self.evict_one_fifo();
                }
            }
        }

        self.inner.insert_atomic(item, self.ttl);

        if !is_update {
            self.live_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn insert_many(&self, items: Vec<T>) {
        for item in items {
            self.insert(item);
        }
    }

    pub fn get_by_id(&self, id: T::Id) -> Option<T> {
        self.maybe_cleanup();

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref_entry) = self.inner.by_id.get(&id) {
                if ref_entry.value().is_expired() {
                    drop(ref_entry);
                    self.inner.remove_atomic(&id);
                    self.live_count.fetch_sub(1, Ordering::Relaxed);
                    self.miss_counter.fetch_add(1, Ordering::Relaxed);
                    None
                } else {
                    self.hit_counter.fetch_add(1, Ordering::Relaxed);
                    Some(ref_entry.value().value.clone())
                }
            } else {
                self.miss_counter.fetch_add(1, Ordering::Relaxed);
                None
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Ok(id_map) = self.inner.by_id.read() {
                if let Some(entry) = id_map.get(&id) {
                    if entry.is_expired() {
                        drop(id_map);
                        self.inner.remove_atomic(&id);
                        self.live_count.fetch_sub(1, Ordering::Relaxed);
                        self.miss_counter.fetch_add(1, Ordering::Relaxed);
                        None
                    } else {
                        self.hit_counter.fetch_add(1, Ordering::Relaxed);
                        Some(entry.value.clone())
                    }
                } else {
                    self.miss_counter.fetch_add(1, Ordering::Relaxed);
                    None
                }
            } else {
                None
            }
        }
    }

    pub fn get_by_slug(&self, slug: &str) -> Option<T> {
        self.maybe_cleanup();

        #[cfg(not(target_arch = "wasm32"))]
        let target_id: Option<T::Id> = self.inner.by_slug.get(slug).map(|r| r.value().clone());

        #[cfg(target_arch = "wasm32")]
        let target_id: Option<T::Id> = self
            .inner
            .by_slug
            .read()
            .ok()
            .and_then(|m| m.get(slug).cloned());

        if let Some(id) = target_id {
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Some(ref_entry) = self.inner.by_id.get(&id) {
                    if ref_entry.value().is_expired() {
                        drop(ref_entry);
                        self.inner.remove_atomic(&id);
                        self.live_count.fetch_sub(1, Ordering::Relaxed);
                        self.miss_counter.fetch_add(1, Ordering::Relaxed);
                        None
                    } else {
                        self.hit_counter.fetch_add(1, Ordering::Relaxed);
                        Some(ref_entry.value().value.clone())
                    }
                } else {
                    self.inner.remove_by_slug_stale(slug);
                    self.miss_counter.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }

            #[cfg(target_arch = "wasm32")]
            {
                if let Ok(id_map) = self.inner.by_id.read() {
                    if let Some(entry) = id_map.get(&id) {
                        if entry.is_expired() {
                            drop(id_map);
                            self.inner.remove_atomic(&id);
                            self.live_count.fetch_sub(1, Ordering::Relaxed);
                            self.miss_counter.fetch_add(1, Ordering::Relaxed);
                            None
                        } else {
                            self.hit_counter.fetch_add(1, Ordering::Relaxed);
                            Some(entry.value.clone())
                        }
                    } else {
                        drop(id_map);
                        self.inner.remove_by_slug_stale(slug);
                        self.miss_counter.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                } else {
                    None
                }
            }
        } else {
            self.miss_counter.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn evict_expired(&self) -> usize {
        let mut count = 0;

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.by_id.retain(|_id, entry| {
                if entry.is_expired() {
                    if let Some(slug) = entry.value.slug() {
                        self.inner.by_slug.remove(slug);
                    }
                    count += 1;
                    false
                } else {
                    true
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let (Ok(mut by_id), Ok(mut by_slug)) =
                (self.inner.by_id.write(), self.inner.by_slug.write())
            {
                let expired_ids: Vec<T::Id> = by_id
                    .iter()
                    .filter(|(_, v)| v.is_expired())
                    .map(|(k, _)| k.clone())
                    .collect();

                count = expired_ids.len();
                for id in &expired_ids {
                    if let Some(entry) = by_id.remove(id) {
                        if let Some(slug) = entry.value.slug() {
                            by_slug.remove(slug);
                        }
                    }
                }
            }
        }

        if count > 0 {
            self.evicted_counter
                .fetch_add(count as u64, Ordering::Relaxed);
            self.live_count.fetch_sub(count, Ordering::Relaxed);
        }
        count
    }

    fn evict_one_fifo(&self) {
        let mut target_id: Option<T::Id> = None;

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(first) = self.inner.by_id.iter().next() {
                target_id = Some(first.key().clone());
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Ok(id_map) = self.inner.by_id.read() {
                if let Some(first) = id_map.keys().next() {
                    target_id = Some(first.clone());
                }
            }
        }

        if let Some(id) = target_id {
            if self.inner.remove_atomic(&id).is_some() {
                self.live_count.fetch_sub(1, Ordering::Relaxed);
                self.evicted_counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn maybe_cleanup(&self) {
        let count = self.access_counter.fetch_add(1, Ordering::Relaxed);
        if self.ttl.is_some() && count % CLEANUP_PROBABILITY == 0 && count > 0 {
            self.evict_expired();
            self.reconcile_count();
        }
    }

    pub fn load(&self, items: Vec<T>) {
        self.clear();
        for item in items {
            self.inner.insert_atomic(item, self.ttl);
        }
        self.reconcile_count();
        if let Ok(mut meta) = self.metadata.write() {
            meta.initialized = true;
            meta.loaded_at = Some(Utc::now());
        }
    }

    pub fn clear(&self) {
        self.inner.clear_all();
        self.live_count.store(0, Ordering::Relaxed);
        self.access_counter.store(0, Ordering::Relaxed);
        self.hit_counter.store(0, Ordering::Relaxed);
        self.miss_counter.store(0, Ordering::Relaxed);
    }

    pub fn len(&self) -> usize {
        self.live_count.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn metadata(&self) -> CacheMetadata {
        let mut meta = self.metadata.read().map(|m| m.clone()).unwrap_or_default();
        meta.count = self.live_count.load(Ordering::Relaxed);
        meta.evicted_count = self.evicted_counter.load(Ordering::Relaxed);
        meta.hit_count = self.hit_counter.load(Ordering::Relaxed);
        meta.miss_count = self.miss_counter.load(Ordering::Relaxed);
        let total = meta.hit_count + meta.miss_count;
        meta.hit_rate = if total == 0 {
            0.0
        } else {
            meta.hit_count as f64 / total as f64
        };
        meta
    }

    fn reconcile_count(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        let actual = self
            .inner
            .by_id
            .iter()
            .filter(|e| !e.value().is_expired())
            .count();

        #[cfg(target_arch = "wasm32")]
        let actual = self
            .inner
            .by_id
            .read()
            .map(|m| m.values().filter(|e| !e.is_expired()).count())
            .unwrap_or(0);

        self.live_count.store(actual, Ordering::Relaxed);
    }
}

impl<T: CacheKey> Default for SvCache<T> {
    fn default() -> Self {
        Self::new()
    }
}
