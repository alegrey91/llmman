//! Content-addressed garbage collection for the blob store and the
//! extracted-model cache.
//!
//! Because the store is fully content-addressed and every tag is just a
//! small pointer file under `manifests/` (see [`crate::storage::oci`]),
//! "what's still needed" can always be recomputed from scratch by reading
//! every *surviving* manifest — there's no refcount or journal to keep in
//! sync (a crash, a manual edit, an interrupted pull can never desync it,
//! because the live set is rebuilt fresh on every sweep).
//!
//! [`referenced_digests`] builds that live set; [`prune_blobs`] and
//! [`prune_cache`] sweep everything not in it. Both are grace-gated the
//! same way [`crate::storage::repair`] gates its stale-temp-file sweep: a
//! blob is written before its manifest/tag pointer, so a blob can be
//! legitimately unreferenced for a moment mid-pull — anything younger than
//! `grace` is left alone. `rm` passes a zero grace (a synchronous,
//! user-initiated removal wants space freed now); the `serve` startup
//! catch-all passes the hour-long window.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, SystemTime};

use anyhow::Context;

use super::OciStore;

/// Grace window for the startup catch-all sweep — matches
/// [`crate::storage::repair::STALE_TMP_FILE_AGE`], so there's one duration
/// to reason about for "how long might a just-written blob be legitimately
/// unreferenced".
pub const GC_GRACE_PERIOD: Duration = Duration::from_secs(60 * 60);

/// What a sweep freed, for reporting.
#[derive(Debug, Default, Clone, Copy)]
pub struct GcStats {
    pub count: usize,
    pub bytes: u64,
}

/// Every blob digest ("sha256:<hex>") still reachable from a surviving
/// tag: each manifest's own digest, its config digest, and every layer
/// digest. Built by walking [`OciStore::list_refs`] and reading each
/// manifest — the same traversal `resolve_model` does per-reference, just
/// over every reference at once. A manifest that can't be read is skipped
/// (its blobs then look unreferenced) — acceptable, since an unreadable
/// manifest is already a broken/incomplete image, but conservative callers
/// run this only right after a successful `remove`.
pub fn referenced_digests(store: &OciStore) -> anyhow::Result<HashSet<String>> {
    let mut live = HashSet::new();
    for desc in store.list_refs() {
        live.insert(desc.digest.clone());
        let Ok(manifest) = store.read_manifest(&desc.digest) else {
            continue;
        };
        live.insert(manifest.config.digest.clone());
        live.extend(manifest.layers.iter().map(|l| l.digest.clone()));
    }
    Ok(live)
}

/// Deletes every blob file under `blobs/sha256/` whose `sha256:<name>`
/// digest isn't in `live`, skipping in-progress temp writes (`tmp-`/
/// `.tmp`, same as [`crate::storage::repair`]) and anything younger than
/// `grace`. A missing blobs directory is a no-op.
pub fn prune_blobs(store_root: &Path, live: &HashSet<String>, grace: Duration) -> anyhow::Result<GcStats> {
    let blobs_dir = store_root.join("blobs").join("sha256");
    let mut stats = GcStats::default();
    let entries = match std::fs::read_dir(&blobs_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
        Err(e) => return Err(e).with_context(|| format!("read {}", blobs_dir.display())),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // In-progress or abandoned write — left to repair's own sweep.
        if name.starts_with("tmp-") || name.ends_with(".tmp") {
            continue;
        }
        if live.contains(&format!("sha256:{name}")) {
            continue;
        }
        if !is_older_than(&path, grace) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!("[llmman] couldn't remove unreferenced blob {name}: {e:#}");
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

/// Deletes every cache subdirectory under `cache_path` whose name (a layer
/// hex for GGUF, a manifest hex for safetensors — see
/// `modelpack::extract_gguf_layer` / `extract_safetensors_dir`) doesn't
/// correspond to a live digest, skipping anything younger than `grace`. A
/// missing cache directory is a no-op.
pub fn prune_cache(cache_path: &Path, live: &HashSet<String>, grace: Duration) -> anyhow::Result<GcStats> {
    let live_hex: HashSet<&str> = live
        .iter()
        .filter_map(|d| d.strip_prefix("sha256:"))
        .collect();
    let mut stats = GcStats::default();
    let entries = match std::fs::read_dir(cache_path) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
        Err(e) => return Err(e).with_context(|| format!("read {}", cache_path.display())),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if live_hex.contains(name) {
            continue;
        }
        if !is_older_than(&path, grace) {
            continue;
        }
        let size = dir_size(&path);
        if let Err(e) = std::fs::remove_dir_all(&path) {
            eprintln!("[llmman] couldn't remove unreferenced cache dir {name}: {e:#}");
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

/// True if `path`'s mtime is at least `grace` old. A zero `grace` makes
/// this always true (used by `rm`, which frees immediately). A file whose
/// mtime can't be read is treated as not-yet-old, so it's left alone
/// rather than deleted on a metadata hiccup.
fn is_older_than(path: &Path, grace: Duration) -> bool {
    if grace.is_zero() {
        return true;
    }
    let Ok(modified) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return false;
    };
    SystemTime::now()
        .duration_since(modified)
        .is_ok_and(|age| age >= grace)
}

/// Total size of the files directly inside `dir` (cache dirs are flat —
/// extracted GGUF/safetensors files, no nesting). Best-effort: unreadable
/// entries contribute 0.
fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Skips both the post-`rm` and startup GC sweeps when `LLMMAN_NOPRUNE` is
/// set to anything other than an explicit falsy value — an escape hatch
/// for shared/read-mostly stores or scripts that `rm` in a loop and would
/// rather prune once at the end themselves. Read fresh at each call site,
/// like every other `LLMMAN_*` var.
pub fn noprune_from_env() -> bool {
    parse_noprune(std::env::var("LLMMAN_NOPRUNE").ok().as_deref())
}

/// Split out from [`noprune_from_env`] for testing without touching the
/// real environment. Unset, blank, or an explicit falsy value
/// (`0`/`false`/`no`/`off`, case-insensitive) all mean "don't skip".
fn parse_noprune(value: Option<&str>) -> bool {
    let Some(v) = value else { return false };
    let v = v.trim();
    if v.is_empty() {
        return false;
    }
    !matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_noprune_is_false_when_unset_blank_or_explicitly_falsy() {
        assert!(!parse_noprune(None));
        assert!(!parse_noprune(Some("")));
        assert!(!parse_noprune(Some("   ")));
        for falsy in ["0", "false", "no", "off", "FALSE", "Off", "  no  "] {
            assert!(!parse_noprune(Some(falsy)), "{falsy:?} should be falsy");
        }
    }

    #[test]
    fn parse_noprune_is_true_for_any_other_value() {
        for truthy in ["1", "true", "yes", "on", "anything"] {
            assert!(parse_noprune(Some(truthy)), "{truthy:?} should be truthy");
        }
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "llmman-gc-test-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn prune_blobs_removes_only_unreferenced_blobs() {
        let root = temp_dir("prune-blobs");
        let blobs = root.join("blobs").join("sha256");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("aaaa"), b"referenced").unwrap();
        std::fs::write(blobs.join("bbbb"), b"orphan").unwrap();
        std::fs::write(blobs.join("tmp-123"), b"in-progress").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_blobs(&root, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert!(blobs.join("aaaa").exists(), "referenced blob must survive");
        assert!(!blobs.join("bbbb").exists(), "orphan blob must be removed");
        assert!(
            blobs.join("tmp-123").exists(),
            "in-progress temp file must be left to repair's sweep"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn prune_cache_removes_only_unreferenced_dirs() {
        let cache = temp_dir("prune-cache");
        std::fs::create_dir_all(cache.join("aaaa")).unwrap();
        std::fs::write(cache.join("aaaa").join("model.gguf"), b"kept").unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        std::fs::write(cache.join("bbbb").join("model.gguf"), b"orphan").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_cache(&cache, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert!(cache.join("aaaa").exists(), "referenced cache dir survives");
        assert!(!cache.join("bbbb").exists(), "orphan cache dir removed");

        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn prune_is_a_no_op_on_missing_directories() {
        let root = temp_dir("missing");
        let live = HashSet::new();
        assert_eq!(prune_blobs(&root, &live, Duration::ZERO).unwrap().count, 0);
        assert_eq!(
            prune_cache(&root.join("cache"), &live, Duration::ZERO)
                .unwrap()
                .count,
            0
        );
    }
}
