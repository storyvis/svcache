//! Regression tests for review findings on 2ff468a. Threads are ordered with channels; the
//! short sleeps only give an already-started call time to sample the clock and block.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use svcache::{CacheKey, ManualClock, SvCache, SvCacheBuilder};

#[derive(Clone, Debug)]
struct Item {
    id: u64,
    slug: Option<String>,
    n: u64,
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

fn item(id: u64, n: u64) -> Item {
    Item { id, slug: None, n }
}

fn slugged(id: u64, slug: &str, n: u64) -> Item {
    Item {
        id,
        slug: Some(slug.into()),
        n,
    }
}

const TTL: Duration = Duration::from_secs(10);
const SETTLE: Duration = Duration::from_millis(150);

fn cache(clock: &ManualClock) -> Arc<SvCache<Item>> {
    let b: SvCacheBuilder<Item> = SvCache::builder();
    Arc::new(b.ttl(TTL).cleanup_budget(0).clock(clock.clone()).build())
}

/// Holds the shard write lock of `id` (via `update`) until the returned sender fires.
fn hold_entry(c: &Arc<SvCache<Item>>, id: u64) -> (Sender<()>, JoinHandle<()>) {
    let (started_tx, started_rx) = channel();
    let (go_tx, go_rx): (Sender<()>, Receiver<()>) = channel();
    let c = c.clone();
    let h = thread::spawn(move || {
        c.update(&id, |_| {
            started_tx.send(()).unwrap();
            go_rx.recv().unwrap();
        })
        .expect("held entry is live");
    });
    started_rx.recv().unwrap();
    (go_tx, h)
}

/// Holds the structural lock (via `get_or_insert_with`'s `make`) until the sender fires.
fn hold_structure(c: &Arc<SvCache<Item>>, id: u64) -> (Sender<()>, JoinHandle<()>) {
    let (started_tx, started_rx) = channel();
    let (go_tx, go_rx): (Sender<()>, Receiver<()>) = channel();
    let c = c.clone();
    let h = thread::spawn(move || {
        c.get_or_insert_with(
            id,
            || {
                started_tx.send(()).unwrap();
                go_rx.recv().unwrap();
                item(id, 0)
            },
            |_| (),
        );
    });
    started_rx.recv().unwrap();
    (go_tx, h)
}

// Finding 2: the clock must be read after the entry lock is acquired.
#[test]
fn get_with_samples_clock_under_entry_lock() {
    let clock = ManualClock::new();
    let c = cache(&clock);
    c.insert(item(1, 0));
    clock.advance(Duration::from_secs(5));
    let (go, holder) = hold_entry(&c, 1);
    let reader = {
        let c = c.clone();
        thread::spawn(move || c.get_with(&1, |v| v.n))
    };
    thread::sleep(SETTLE);
    clock.advance(Duration::from_secs(20)); // entry expired at t=10
    go.send(()).unwrap();
    holder.join().unwrap();
    assert_eq!(
        reader.join().unwrap(),
        None,
        "expired entry returned after waiting"
    );
}

#[test]
fn get_or_insert_with_existing_samples_clock_under_entry_lock() {
    let clock = ManualClock::new();
    let c = cache(&clock);
    c.insert(item(1, 0));
    clock.advance(Duration::from_secs(5));
    let (go, holder) = hold_entry(&c, 1);
    let caller = {
        let c = c.clone();
        thread::spawn(move || c.get_or_insert_with(1, || item(1, 99), |v| v.n))
    };
    thread::sleep(SETTLE);
    clock.advance(Duration::from_secs(20));
    go.send(()).unwrap();
    holder.join().unwrap();
    assert_eq!(
        caller.join().unwrap(),
        99,
        "expired entry reused instead of replaced"
    );
}

#[test]
fn get_or_insert_with_stamps_expiry_after_make_and_f() {
    let clock = ManualClock::new();
    let c = cache(&clock);
    let slow = clock.clone();
    c.get_or_insert_with(
        1,
        || {
            slow.advance(Duration::from_secs(20));
            item(1, 0)
        },
        |_| (),
    );
    assert!(
        c.get_with(&1, |_| ()).is_some(),
        "fresh entry expired at once (slow make)"
    );

    let slow = clock.clone();
    c.get_or_insert_with(2, || item(2, 0), |_| slow.advance(Duration::from_secs(20)));
    assert!(
        c.get_with(&2, |_| ()).is_some(),
        "fresh entry expired at once (slow f)"
    );
}

#[test]
fn insert_waiting_for_lock_stamps_fresh_expiry() {
    let clock = ManualClock::new();
    let c = cache(&clock);
    let (go, holder) = hold_structure(&c, 2);
    let writer = {
        let c = c.clone();
        thread::spawn(move || c.insert_many(vec![item(1, 0), item(3, 0)]))
    };
    thread::sleep(SETTLE);
    clock.advance(Duration::from_secs(20));
    go.send(()).unwrap();
    holder.join().unwrap();
    writer.join().unwrap();
    assert!(
        c.get_with(&1, |_| ()).is_some(),
        "insert stamped with pre-wait time"
    );
    assert!(c.get_with(&3, |_| ()).is_some());
}

// Finding 3: replacing an existing id with an unchanged slug must not need the structural lock.
#[test]
fn replacement_does_not_take_structural_lock() {
    let clock = ManualClock::new();
    let c = cache(&clock);
    c.insert(slugged(1, "a", 0));
    c.insert(item(3, 0));
    let (go, holder) = hold_structure(&c, 2);
    let (done_tx, done_rx) = channel();
    let writer = {
        let c = c.clone();
        thread::spawn(move || {
            c.insert(slugged(1, "a", 7));
            c.insert_many(vec![item(3, 8)]);
            done_tx.send(()).unwrap();
        })
    };
    let finished = done_rx.recv_timeout(Duration::from_secs(2)).is_ok();
    go.send(()).unwrap();
    holder.join().unwrap();
    writer.join().unwrap();
    assert!(finished, "replacement blocked on the structural lock");
    assert_eq!(c.get_by_slug("a").map(|v| v.n), Some(7));
    assert_eq!(c.get_with(&3, |v| v.n), Some(8));
}

#[test]
fn replacement_reclaims_shared_slug() {
    let c = SvCache::new();
    c.insert(slugged(1, "s", 0));
    c.insert(slugged(2, "s", 0));
    c.insert(slugged(1, "s", 1));
    assert_eq!(c.get_by_slug("s").map(|v| v.id), Some(1));
}

#[test]
fn replacement_keeps_fixed_expiry_reset() {
    let clock = ManualClock::new();
    let c = cache(&clock);
    c.insert(slugged(1, "a", 0));
    clock.advance(Duration::from_secs(8));
    c.insert(slugged(1, "a", 1));
    clock.advance(Duration::from_secs(8));
    assert_eq!(c.get_by_slug("a").map(|v| v.n), Some(1));
}
