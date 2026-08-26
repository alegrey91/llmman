use std::time::Duration;

use clap::Args;

use crate::storage::{gc, OciStore};

#[derive(Args, Debug)]
pub struct RmArgs {
    /// Reference(s) to remove (e.g. registry.example.com/mymodel:latest)
    #[arg(value_name = "REFERENCE", required = true, num_args = 1..)]
    pub references: Vec<String>,
}

pub fn run(args: &RmArgs) -> anyhow::Result<()> {
    let store_root = crate::default_store()?;
    let store = OciStore::open(&store_root)?;

    let mut any_err = false;
    let mut any_removed = false;
    for raw in &args.references {
        // resolve_ollama_api, not resolve: a bare name pulled via the
        // Ollama API (POST /api/pull, /api/chat, ...) is stored under
        // docker.io/ai/<name>, not hf.co/<name> — must resolve the same
        // way here or `llmman rm <bare-name>` looks for the wrong entry.
        let reference = crate::shortnames::resolve_ollama_api(raw);
        match store.remove(&reference) {
            Ok(()) => {
                println!("Removed {}", reference);
                any_removed = true;
            }
            Err(e) => {
                eprintln!("Error removing {}: {}", reference, e);
                any_err = true;
            }
        }
    }

    // GC once, after every requested reference is untagged: recompute the
    // still-referenced digest set from the surviving manifests, then sweep
    // blobs/cache not in it. Grace 0 — `rm` is synchronous and
    // user-initiated, so free the space immediately (the startup sweep in
    // `serve` uses the grace window for the concurrent-pull case instead).
    if any_removed && !gc::noprune_from_env() {
        let live = gc::referenced_digests(&store)?;
        let cache_path = crate::default_cache()?;
        let blob_stats = gc::prune_blobs(&store_root, &live, Duration::ZERO)?;
        let cache_stats = gc::prune_cache(&cache_path, &live, Duration::ZERO)?;
        if blob_stats.count > 0 || cache_stats.count > 0 {
            println!(
                "Freed {} ({} blobs, {} cache entries)",
                crate::fmt::human_size(blob_stats.bytes + cache_stats.bytes),
                blob_stats.count,
                cache_stats.count
            );
        }
    }

    if any_err {
        anyhow::bail!("one or more removals failed");
    }
    Ok(())
}
