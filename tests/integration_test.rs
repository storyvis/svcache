use std::time::Duration;
use svcache::{CacheKey, SvCache};

/// Test struct implementing CacheKey
#[derive(Clone, Debug, PartialEq)]
struct User {
    id: u64,
    name: String,
    slug: Option<String>,
}

impl CacheKey for User {
    type Id = u64;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn slug(&self) -> Option<&str> {
        self.slug.as_deref()
    }
}

fn make_user(id: u64, name: &str, slug: Option<&str>) -> User {
    User {
        id,
        name: name.to_string(),
        slug: slug.map(|s| s.to_string()),
    }
}

#[test]
fn test_new_cache_is_empty() {
    let cache: SvCache<User> = SvCache::new();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_insert_and_get_by_id() {
    let cache = SvCache::new();
    let user = make_user(1, "Alice", None);

    cache.insert(user.clone());

    assert_eq!(cache.len(), 1);
    let retrieved = cache.get_by_id(1).unwrap();
    assert_eq!(retrieved.id, 1);
    assert_eq!(retrieved.name, "Alice");
}

#[test]
fn test_insert_and_get_by_slug() {
    let cache = SvCache::new();
    let user = make_user(1, "Alice", Some("alice"));

    cache.insert(user.clone());

    let retrieved = cache.get_by_slug("alice").unwrap();
    assert_eq!(retrieved.id, 1);
    assert_eq!(retrieved.name, "Alice");
}

#[test]
fn test_get_by_slug_nonexistent_returns_none() {
    let cache: SvCache<User> = SvCache::new();
    assert!(cache.get_by_slug("nonexistent").is_none());
}

#[test]
fn test_get_by_id_nonexistent_returns_none() {
    let cache: SvCache<User> = SvCache::new();
    assert!(cache.get_by_id(999).is_none());
}

#[test]
fn test_insert_many() {
    let cache = SvCache::new();
    let users = vec![
        make_user(1, "Alice", Some("alice")),
        make_user(2, "Bob", Some("bob")),
        make_user(3, "Charlie", Some("charlie")),
    ];

    cache.insert_many(users);

    assert_eq!(cache.len(), 3);
    assert!(cache.get_by_id(1).is_some());
    assert!(cache.get_by_id(2).is_some());
    assert!(cache.get_by_id(3).is_some());
    assert!(cache.get_by_slug("alice").is_some());
    assert!(cache.get_by_slug("bob").is_some());
    assert!(cache.get_by_slug("charlie").is_some());
}

#[test]
fn test_update_existing_entry() {
    let cache = SvCache::new();
    let user1 = make_user(1, "Alice", Some("alice"));
    let user1_updated = make_user(1, "Alice Updated", Some("alice"));

    cache.insert(user1);
    assert_eq!(cache.len(), 1);

    cache.insert(user1_updated);
    // Length should not increase on update
    assert_eq!(cache.len(), 1);

    let retrieved = cache.get_by_id(1).unwrap();
    assert_eq!(retrieved.name, "Alice Updated");
}

#[test]
fn test_clear() {
    let cache = SvCache::new();
    cache.insert(make_user(1, "Alice", Some("alice")));
    cache.insert(make_user(2, "Bob", Some("bob")));

    assert_eq!(cache.len(), 2);

    cache.clear();

    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert!(cache.get_by_id(1).is_none());
    assert!(cache.get_by_slug("alice").is_none());
}

#[test]
fn test_load_replaces_all() {
    let cache = SvCache::new();
    cache.insert(make_user(1, "Alice", Some("alice")));
    cache.insert(make_user(2, "Bob", Some("bob")));

    let new_items = vec![
        make_user(10, "Xavier", Some("xavier")),
        make_user(11, "Yolanda", Some("yolanda")),
    ];

    cache.load(new_items);

    assert_eq!(cache.len(), 2);
    assert!(cache.get_by_id(1).is_none());
    assert!(cache.get_by_id(10).is_some());
    assert!(cache.get_by_slug("xavier").is_some());
}

#[test]
fn test_with_limit_evicts() {
    let cache = SvCache::with_limit(2);

    cache.insert(make_user(1, "Alice", None));
    cache.insert(make_user(2, "Bob", None));
    assert_eq!(cache.len(), 2);

    // Inserting a third should evict one (FIFO)
    cache.insert(make_user(3, "Charlie", None));
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_ttl_expiry() {
    // Use a very short TTL
    let cache = SvCache::with_ttl(Duration::from_millis(50));

    cache.insert(make_user(1, "Alice", Some("alice")));
    assert!(cache.get_by_id(1).is_some());

    // Wait for expiry
    std::thread::sleep(Duration::from_millis(100));

    // After expiry, get should return None
    assert!(cache.get_by_id(1).is_none());
}

#[test]
fn test_evict_expired() {
    let cache = SvCache::with_ttl(Duration::from_millis(50));

    cache.insert(make_user(1, "Alice", None));
    cache.insert(make_user(2, "Bob", None));

    std::thread::sleep(Duration::from_millis(100));

    let evicted = cache.evict_expired();
    assert_eq!(evicted, 2);
    assert_eq!(cache.len(), 0);
}

#[cfg(feature = "stats")]
#[test]
fn test_metadata() {
    let cache = SvCache::new();
    cache.insert(make_user(1, "Alice", None));
    cache.insert(make_user(2, "Bob", None));

    // Trigger a hit
    cache.get_by_id(1);
    // Trigger a miss
    cache.get_by_id(999);

    let meta = cache.metadata();
    assert_eq!(meta.count, 2);
    assert_eq!(meta.hit_count, 1);
    assert_eq!(meta.miss_count, 1);
    assert!((meta.hit_rate - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_metadata_after_load() {
    let cache = SvCache::new();
    let items = vec![make_user(1, "Alice", None), make_user(2, "Bob", None)];

    cache.load(items);

    let meta = cache.metadata();
    assert!(meta.initialized);
    assert!(meta.loaded_at.is_some());
    assert_eq!(meta.count, 2);
}

#[test]
fn test_default_trait() {
    let cache: SvCache<User> = SvCache::default();
    assert!(cache.is_empty());
}

#[test]
fn test_slug_without_slug_field() {
    // User without slug should not be findable by slug
    let cache = SvCache::new();
    let user = make_user(1, "Alice", None);
    cache.insert(user);

    assert!(cache.get_by_slug("alice").is_none());
    assert!(cache.get_by_id(1).is_some());
}
