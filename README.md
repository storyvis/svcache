# svcache

**Isomorphic Dual-Index Cache for WASM and Native Runtimes**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`svcache` automatically toggles between a sharded `DashMap` architecture on native multi-threaded targets (like Tokio) and a lean `RwLock<HashMap>` structure on single-threaded WASM environments (like Cloudflare Workers).

## Features

- **Dual-index lookup** — Primary key (ID) and optional secondary slug/name index
- **Isomorphic** — Single API that compiles for both native and `wasm32` targets
- **TTL support** — Monotonic-clock expiry, fixed or sliding, with bounded incremental cleanup
- **Max entry cap** — SIEVE eviction (exact limit, O(1) amortized)
- **In-place access** — `get_with`, `update`, `get_or_insert_with` without cloning
- **Eviction callback** — `on_evict` with the reason (expired, capacity, removed, cleared)
- **Cache metadata** — Hit/miss counters (striped per thread), hit rate, eviction count, timestamps
- **Zero-config** — Works out of the box with `SvCache::new()`

## Architecture

| Target | Engine | Thread Safety |
|--------|--------|---------------|
| Native (x86_64, aarch64, etc.) | `DashMap` (sharded concurrent map) | Sharded read/write locks |
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

// With max entries (SIEVE eviction)
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

### Builder and in-place access

```rust
use svcache::{EvictReason, SvCache, TtlMode};
use std::time::Duration;

let sessions = SvCache::<User>::builder()
    .ttl(Duration::from_secs(60))
    .ttl_mode(TtlMode::Sliding)        // every hit/update pushes expiry to now + ttl
    .max_entries(16_384)
    .cleanup_budget(0)                 // no inline sweeps; a background task sweeps instead
    .on_evict(|id, _user, reason| {
        if reason == EvictReason::Expired {
            println!("session {id} expired");
        }
    })
    .build();

// Read without cloning.
let name_len = sessions.get_with(&1, |u| u.username.len());

// Mutate in place (no clone, no lost updates).
sessions.update(&1, |u| u.username.push('!'));

// Create on first use, then mutate atomically.
sessions.get_or_insert_with(2, || User { id: 2, username: "bob".into() }, |u| u.username.len());

sessions.touch(&2);          // refresh expiry explicitly
sessions.remove(&2);         // also drops the slug mapping

// Off the hot path, e.g. every 100 ms:
sessions.evict_expired_budget(256);
```

## API

| Method | Description |
|--------|-------------|
| `SvCache::new()` | Create unbounded cache |
| `SvCache::with_ttl(duration)` | Cache with time-based expiry |
| `SvCache::with_limit(max)` | Cache with max entry count |
| `SvCache::with_ttl_and_limit(duration, max)` | Both TTL and limit |
| `SvCache::builder()` | `ttl`, `max_entries`, `ttl_mode`, `cleanup_interval`, `cleanup_budget`, `clock`, `on_evict` |
| `insert(item)` | Insert or update a single item |
| `insert_many(items)` | Insert multiple items |
| `get_by_id(id)` | Lookup by primary key (clones) |
| `get_by_slug(slug)` | Lookup by secondary slug (clones) |
| `get_with(&id, f)` | Run `f(&T)` on the entry, no clone |
| `update(&id, f)` | Run `f(&mut T)` on the entry in place |
| `get_or_insert_with(id, make, f)` | Insert `make()` if absent, then run `f(&mut T)` |
| `touch(&id)` | Mark accessed and refresh expiry |
| `remove(&id)` | Remove and return the entry |
| `load(items)` | Replace contents with `items` (no empty window) |
| `clear()` | Remove all entries and reset counters |
| `evict_expired()` | Full expiry sweep |
| `evict_expired_budget(n)` | Incremental sweep of at most `n` entries |
| `len()` / `is_empty()` | Entry count (includes expired entries not yet reclaimed) |
| `metadata()` | Get cache statistics |

## Semantics

- **Eviction policy — SIEVE.** Each entry has a `visited` bit, set by any successful lookup,
  update or re-insert. At capacity, a hand walks from the oldest entry towards the newest,
  clearing set bits and evicting the first entry whose bit is clear; expired entries are
  evicted regardless. New entries start unvisited. Without hits this is exact FIFO. The
  entry limit is exact, also under concurrent inserts.
- **Zero capacity.** `max_entries(0)` disables retention. Inserts and loads discard their
  inputs without eviction callbacks because no entry is installed. `get_or_insert_with`
  still runs `make` and `f` and returns their result, but does not retain the value.
- **Clock.** Monotonic: `std::time::Instant` natively, `performance.now()` via
  [`web-time`](https://crates.io/crates/web-time) on `wasm32`. Millisecond resolution; an
  entry is expired once more than `ttl` has passed. `ManualClock` can be injected for tests;
  advancing past its `u64` millisecond range saturates rather than wrapping backwards.
- **TTL mode.** `Fixed` (default): expiry is set on insert/re-insert. `Sliding`: every
  successful lookup or `update` also moves expiry to `now + ttl`. `touch` refreshes in both.
- **Cleanup.** Expired entries are never returned. They are reclaimed when a lookup finds
  them, when the eviction hand reaches them, by an inline sweep of at most `cleanup_budget`
  entries (default 64) that runs at most once per `cleanup_interval` (default
  `min(ttl / 4, 1 s)`), and by `evict_expired_budget` / `evict_expired`.
  Sweeps follow live entries, so removed slab slots do not consume the budget after a
  cache shrinks. A due sweep runs before capacity eviction on new-id insertion; it remains
  interval- and budget-limited, so it does not promise to find every expired entry before
  evicting a live one. A failed nonblocking lock attempt leaves cleanup due for the next
  request. Slab storage may still reflect the cache's historical peak until emptied.
- **Indices.** Each entry stores the slug it is indexed under. A slug lookup only hits if the
  entry still carries that slug and is live; every removal path also removes the slug
  mapping (if it still points at that id). If two ids share a slug, the last insert owns it.
- **Concurrency.** Lookups, `update`, `touch`, and `insert` of an existing id whose slug is
  unchanged do not need the structural lock. Replacements hold the entry's shard lock and,
  for slugged entries, a slug read lock to protect ownership. Inserts of new ids, slug changes,
  removals, eviction, sweeps, `load` and `clear` serialize on one mutex. The clock is read after the
  relevant lock is acquired, and expiry only ever moves forward under concurrent refreshes. Closures
  passed to `get_with` / `update` / `get_or_insert_with` run under a shard lock and must not
  call back into the cache. `on_evict` runs after the entry is gone from both indices with
  no lock held, and may call back into the cache.
- **`load`.** Upserts the new items, then removes ids that are not among them. An id present
  before and after never misses during the load, but readers may observe a mix of old and
  new values across different ids.

## Cargo features

| Feature | Default | Effect |
|---------|---------|--------|
| `stats` | on | Hit/miss counters (striped across cache lines). Disable to skip the per-lookup increment; `metadata()` then reports 0 hits/misses. |

## Reproducing expiry measurements

`cargo run --release --locked --example expiry_bench` emits five trials of CSV for sparse
and dense cleanup, a fixed-TTL retention trace, mixed hot/scan SIEVE requests, and capacity
churn. Copy the same example into the baseline checkout and run both builds on a quiet
host. The request sequences and hit counts are deterministic; elapsed times are measurements,
not pass/fail thresholds. The small TTL trace isolates expiry/admission ordering and does
not represent an application workload. The original `bench` example also exercises threaded
lookups and replacements.

Measured on 2026-11-05 against `90fcbad` with the same example and separate release target
directories: Apple M4 Pro (12 logical CPUs), macOS 26.5.1, Rust 1.95.0, default `stats`
feature. Three alternating runs per revision contain five trials each (15 samples below).
Builds were paused; an unrelated Slurm controller used about one CPU and desktop/API
processes remained active. These are synthetic measurements, not isolated-host guarantees.

| Trace | Baseline | Updated |
|---|---:|---:|
| Reclaim 64 expired entries after a 32,768-entry peak, budget 64 | 512 calls | 1 call |
| Same sparse cleanup, median elapsed | 24.292 µs | 1.958 µs |
| Reclaim 32,768 expired entries, budget 64 | 512 calls | 512 calls |
| Same dense cleanup, median elapsed | 1,023.291 µs | 996.250 µs |
| TTL-pressure probe hits (purpose-built ordering regression) | 0 / 1,000 | 1,000 / 1,000 |
| Mixed hot/scan SIEVE hits | 179,936 / 200,000 | 179,936 / 200,000 |

The existing threaded `bench` example ran once per alternating run. Median `get_by_id`
throughput was 54.6 → 54.7 Mops/s; single-thread TTL capacity insertion was 316.6 →
325.2 ns/op (about 2.7% slower). These do not establish a general workload speedup.
The normal hit path is unchanged until cleanup is due; a failed cleanup lock attempt
now remains due and can cause further nonblocking probes during sustained contention.

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
