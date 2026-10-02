//! Deterministic interleavings for the follow-up review. A key's hash can pause
//! immediately before a map lock, and ManualClock advances time without sleeps.

use std::{
    cell::Cell,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};
use svcache::{CacheKey, ManualClock, SvCache};

thread_local! {
    static BLOCK_KEY: Cell<Option<u64>> = const { Cell::new(None) };
    static SKIP_MATCHES: Cell<usize> = const { Cell::new(0) };
    // (id() calls before arming, key to pause, matching hashes to skip)
    static ARM_ON_ID: Cell<Option<(usize, u64, usize)>> = const { Cell::new(None) };
}

struct Gate {
    reached: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}
#[derive(Clone)]
struct Key {
    id: u64,
    gate: Arc<Gate>,
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Key {}
impl Hash for Key {
    fn hash<H: Hasher>(&self, h: &mut H) {
        let block = BLOCK_KEY.with(|b| {
            if b.get() != Some(self.id) {
                return false;
            }
            let skip = SKIP_MATCHES.with(|s| {
                let n = s.get();
                s.set(n.saturating_sub(1));
                n > 0
            });
            if skip {
                return false;
            }
            b.set(None);
            true
        });
        if block {
            self.gate.reached.send(()).unwrap();
            self.gate.release.lock().unwrap().recv().unwrap();
        }
        self.id.hash(h);
    }
}
#[derive(Clone)]
struct Item {
    key: Key,
    slug: Option<String>,
    value: u64,
}
impl CacheKey for Item {
    type Id = Key;
    fn id(&self) -> Key {
        ARM_ON_ID.with(|arm| {
            if let Some((remaining, key, skip)) = arm.get() {
                if remaining == 1 {
                    arm.set(None);
                    BLOCK_KEY.with(|b| b.set(Some(key)));
                    SKIP_MATCHES.with(|s| s.set(skip));
                } else {
                    arm.set(Some((remaining - 1, key, skip)));
                }
            }
        });
        self.key.clone()
    }
    fn slug(&self) -> Option<&str> {
        self.slug.as_deref()
    }
}
fn gate() -> (Arc<Gate>, mpsc::Receiver<()>, mpsc::Sender<()>) {
    let (tx, rx) = mpsc::channel();
    let (go_tx, go_rx) = mpsc::channel();
    (
        Arc::new(Gate {
            reached: tx,
            release: Mutex::new(go_rx),
        }),
        rx,
        go_tx,
    )
}
fn item(g: &Arc<Gate>, id: u64, slug: Option<&str>, value: u64) -> Item {
    Item {
        key: Key {
            id,
            gate: g.clone(),
        },
        slug: slug.map(str::to_owned),
        value,
    }
}
#[test]
fn replacement_cannot_use_obsolete_slug_ownership() {
    let (g, reached, release) = gate();
    let cache = Arc::new(SvCache::new());
    cache.insert(item(&g, 1, Some("shared"), 0));
    let (c, g2) = (cache.clone(), g.clone());
    let writer = std::thread::spawn(move || {
        BLOCK_KEY.with(|b| b.set(Some(1)));
        c.insert(item(&g2, 1, Some("shared"), 1));
    });
    // Pause before the entry lock. The old implementation already sampled
    // slug ownership here, allowing an intervening owner change to go unnoticed.
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    cache.insert(item(&g, 2, Some("shared"), 0));
    // This read forces writer 1's replacement to linearize after insert 2:
    // the replacement cannot have taken effect yet because we still read value 0.
    assert_eq!(
        cache
            .get_by_id(Key {
                id: 1,
                gate: g.clone()
            })
            .unwrap()
            .value,
        0
    );
    release.send(()).unwrap();
    writer.join().unwrap();
    assert_eq!(
        cache.get_by_slug("shared").unwrap().key.id,
        1,
        "replacement completed after the intervening read but did not reclaim its slug"
    );
}
#[test]
fn slug_change_stamps_expiry_after_entry_lock_wait() {
    let (g, reached, release) = gate();
    let clock = ManualClock::new();
    let cache = Arc::new(
        SvCache::builder()
            .ttl(Duration::from_secs(10))
            .cleanup_budget(0)
            .clock(clock.clone())
            .build(),
    );
    cache.insert(item(&g, 1, Some("old"), 0));
    let (c, g2) = (cache.clone(), g.clone());
    let writer = std::thread::spawn(move || {
        // Arm in insert_locked's id() call, after the initial fast-path attempt.
        ARM_ON_ID.with(|a| a.set(Some((2, 1, 0))));
        c.insert(item(&g2, 1, Some("new"), 1));
    });
    // Pause the structural replacement before acquiring the entry's shard.
    // Expiry must be sampled after this wait, not just after acquiring order.
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    clock.advance(Duration::from_secs(11));
    release.send(()).unwrap();
    writer.join().unwrap();
    assert!(
        cache.get_by_slug("new").is_some(),
        "replacement expired before installation"
    );
}
#[test]
fn capacity_insert_stamps_expiry_after_eviction_wait() {
    let (g, reached, release) = gate();
    let clock = ManualClock::new();
    let cache = Arc::new(
        SvCache::builder()
            .ttl(Duration::from_secs(10))
            .max_entries(1)
            .cleanup_budget(0)
            .clock(clock.clone())
            .build(),
    );
    cache.insert(item(&g, 1, None, 0));
    let (c, g2) = (cache.clone(), g.clone());
    let writer = std::thread::spawn(move || {
        BLOCK_KEY.with(|b| b.set(Some(1)));
        c.insert(item(&g2, 2, None, 0));
    });
    // New key 2 is absent. Pause on victim 1, after expiry for key 2 was computed.
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    clock.advance(Duration::from_secs(11));
    release.send(()).unwrap();
    writer.join().unwrap();
    assert!(
        cache.get_by_id(Key { id: 2, gate: g }).is_some(),
        "new entry expired during eviction"
    );
}

#[test]
fn get_or_insert_must_not_overwrite_a_concurrently_revived_entry() {
    let (g, reached, release) = gate();
    let clock = ManualClock::new();
    let cache = Arc::new(
        SvCache::builder()
            .ttl(Duration::from_secs(10))
            .cleanup_budget(0)
            .clock(clock.clone())
            .build(),
    );
    cache.insert(item(&g, 1, None, 0));
    clock.advance(Duration::from_secs(11));
    let (c, g2) = (cache.clone(), g.clone());
    let caller = std::thread::spawn(move || {
        BLOCK_KEY.with(|b| b.set(Some(1)));
        SKIP_MATCHES.with(|s| s.set(2));
        c.get_or_insert_with(
            Key {
                id: 1,
                gate: g2.clone(),
            },
            || item(&g2, 1, None, 0),
            |v| {
                v.value += 1;
                v.value
            },
        )
    });
    // Both expired checks have completed; pause before the conditional removal.
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    cache.insert(item(&g, 1, None, 41));
    release.send(()).unwrap();
    let returned = caller.join().unwrap();
    let final_value = cache.get_by_id(Key { id: 1, gate: g }).unwrap().value;
    // The two possible serial orders are get_or_insert then insert (1,41),
    // or insert then get_or_insert (42,42). (1,1) loses the live replacement.
    assert!(
        matches!((returned, final_value), (1, 41) | (42, 42)),
        "non-atomic result: returned={returned}, final={final_value}"
    );
}

#[test]
fn new_entry_stamps_expiry_after_destination_lock_wait() {
    let (g, reached, release) = gate();
    let clock = ManualClock::new();
    let cache = Arc::new(
        SvCache::builder()
            .ttl(Duration::from_secs(10))
            .cleanup_budget(0)
            .clock(clock.clone())
            .build(),
    );
    let (c, g2) = (cache.clone(), g.clone());
    let writer = std::thread::spawn(move || {
        // In insert_locked, skip the absent-entry check and pause the actual insertion.
        ARM_ON_ID.with(|a| a.set(Some((2, 2, 1))));
        c.insert(item(&g2, 2, None, 0));
    });
    reached.recv_timeout(Duration::from_secs(5)).unwrap();
    clock.advance(Duration::from_secs(11));
    release.send(()).unwrap();
    writer.join().unwrap();
    assert!(
        cache.get_by_id(Key { id: 2, gate: g }).is_some(),
        "new entry expired waiting for its destination shard"
    );
}
