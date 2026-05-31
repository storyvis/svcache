# svcache

**Isomorphic Dual-Index Cache for WASM and Native Runtimes**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`svcache` automatically toggles between a lock-free, sharded `DashMap` architecture on native multi-threaded targets (like Tokio) and a lean `RwLock<HashMap>` structure on single-threaded WASM environments (like Cloudflare Workers).

## Features

- **Dual-index lookup** — Primary key (ID) and optional secondary slug/name index
- **Isomorphic** — Single API that compiles for both native and `wasm32` targets
- **TTL support** — Automatic time-based expiry with lazy + periodic cleanup
- **Max entry cap** — FIFO eviction when a configurable limit is reached
- **Cache metadata** — Hit/miss counters, hit rate, eviction count, timestamps
- **Zero-config** — Works out of the box with `SvCache::new()`

## Architecture

| Target | Engine | Thread Safety |
|--------|--------|---------------|
| Native (x86_64, aarch64, etc.) | `DashMap` (sharded concurrent map) | Lock-free reads, sharded writes |
| WASM (`wasm32`) | `RwLock<HashMap>` | Single-threaded safe |

## Quick Start

```rust
use svcache::{CacheKey, SvCache};

#[derive(Clone)]
struct User {
    id: u64,
    username: String,
}

impl CacheKey for User {
    type Id = u64;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn slug(&self) -> Option<&str> {
        Some(&self.username)
    }
}

// Unbounded cache
let cache = SvCache::new();

// With TTL (entries expire after 5 minutes)
let cache = SvCache::with_ttl(std::time::Duration::from_secs(300));

// With max entries (FIFO eviction)
let cache = SvCache::with_limit(1000);

// With both
let cache = SvCache::with_ttl_and_limit(
    std::time::Duration::from_secs(300),
    1000,
);

// Insert
cache.insert(User { id: 1, username: "alice".into() });

// Lookup by ID
let user = cache.get_by_id(1);

// Lookup by slug
let user = cache.get_by_slug("alice");

// Bulk load (replaces all entries)
cache.load(vec![
    User { id: 1, username: "alice".into() },
    User { id: 2, username: "bob".into() },
]);

// Metadata
let meta = cache.metadata();
println!("Entries: {}, Hit rate: {:.1}%", meta.count, meta.hit_rate * 100.0);
```

## API

| Method | Description |
|--------|-------------|
| `SvCache::new()` | Create unbounded cache |
| `SvCache::with_ttl(duration)` | Cache with time-based expiry |
| `SvCache::with_limit(max)` | Cache with max entry count |
| `SvCache::with_ttl_and_limit(duration, max)` | Both TTL and limit |
| `insert(item)` | Insert or update a single item |
| `insert_many(items)` | Insert multiple items |
| `get_by_id(id)` | Lookup by primary key |
| `get_by_slug(slug)` | Lookup by secondary slug |
| `load(items)` | Clear and bulk-load items |
| `clear()` | Remove all entries and reset counters |
| `evict_expired()` | Manually trigger expiry sweep |
| `len()` / `is_empty()` | Entry count |
| `metadata()` | Get cache statistics |

## Implementing `CacheKey`

Any type you want to cache must implement the `CacheKey` trait:

```rust
pub trait CacheKey: Clone + Send + Sync + 'static {
    type Id: Hash + Eq + Clone + Send + Sync + 'static;

    fn id(&self) -> Self::Id;

    /// Optional — return None if no slug lookup is needed
    fn slug(&self) -> Option<&str> {
        None
    }
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
