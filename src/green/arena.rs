use std::sync::Mutex;

use bumpalo::Bump;

pub(crate) struct GreenArena {
    /// The lock is held on every addition to the arena (can happen after building
    /// due to mutable trees). Reading doesn't acquire a lock.
    write_lock: Mutex<()>,

    arena: Bump,
}

// SAFETY: We only mutate when having a lock, and mutating doesn't invalidate existing pointers.
unsafe impl Sync for GreenArena {}

impl GreenArena {}
