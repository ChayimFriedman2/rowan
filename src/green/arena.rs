use std::fmt;
use std::sync::Mutex;

use bumpalo::Bump;
use rustc_hash::FxHashSet;
use text_size::TextSize;
use triomphe::{Arc, UniqueArc};

use crate::SyntaxKind;
use crate::green::GreenChild;
use crate::green::node::{GreenNodeHead, GreenNodeInTree};
use crate::green::token::{GreenTokenHead, GreenTokenInTree};
use crate::utility_types::PtrEqArc;

pub(crate) struct GreenTree {
    arena: Bump,

    /// Here we put all trees that our nodes may refer to; they must be kept alive as long as we are.
    ///
    /// Dependencies are never cleared, only on drop - because we have no way to tell if we still
    /// depend on it.
    ///
    /// If you really want, it should be possible to create a cyclic dependency between green trees -
    /// but if you try so hard, it will definitely be your own problem. Generally, green tree dependencies
    /// form a tree, and it is very small and short-lived, except for the root node (we add dependencies
    /// mostly when editing trees).
    dependencies: Mutex<FxHashSet<PtrEqArc<GreenTree>>>,
}

// SAFETY: We only mutate when having mutable access, and mutating doesn't invalidate existing pointers.
unsafe impl Sync for GreenTree {}

impl GreenTree {
    // This needs to be inside `UniqueArc` because otherwise the `verify_origin()` comparisons
    // are messed up.
    #[inline]
    pub(crate) fn new() -> UniqueArc<Self> {
        UniqueArc::new(Self { arena: Bump::new(), dependencies: Mutex::new(FxHashSet::default()) })
    }

    #[inline]
    pub(super) fn alloc_token(&mut self, kind: SyntaxKind, text: &str) -> GreenTokenInTree {
        // SAFETY: We have mutable access.
        unsafe { self.alloc_token_unchecked(kind, text) }
    }

    /// # Safety
    ///
    /// You must ensure there is no concurrent allocation.
    pub(crate) unsafe fn alloc_token_unchecked(
        &self,
        kind: SyntaxKind,
        text: &str,
    ) -> GreenTokenInTree {
        let layout = GreenTokenHead::layout(text.len());
        let token = self.arena.alloc_layout(layout);
        let token = GreenTokenInTree { data: token.cast() };
        // SAFETY: The token is allocated, we don't need it to be initialized for the writing.
        unsafe {
            token.header_ptr_mut().write(GreenTokenHead::new(kind, text));
            token.text_ptr_mut().copy_from_nonoverlapping(text.as_bytes().as_ptr(), text.len());
        }
        token
    }

    #[inline]
    pub(super) fn alloc_node(
        &mut self,
        kind: SyntaxKind,
        text_len: TextSize,
        children_len: usize,
        children: impl Iterator<Item = GreenChild>,
    ) -> GreenNodeInTree {
        // SAFETY: We have mutable access.
        unsafe { self.alloc_node_unchecked(kind, text_len, children_len, children) }
    }

    /// # Safety
    ///
    /// You must ensure there is no concurrent allocation.
    #[inline]
    pub(crate) unsafe fn alloc_node_unchecked(
        &self,
        kind: SyntaxKind,
        text_len: TextSize,
        children_len: usize,
        mut children: impl Iterator<Item = GreenChild>,
    ) -> GreenNodeInTree {
        let layout = GreenNodeHead::layout(children_len);
        let token = self.arena.alloc_layout(layout);
        let node = GreenNodeInTree { data: token.cast() };
        // SAFETY: The node is allocated, we don't need it to be initialized for the writing.
        unsafe {
            node.header_ptr_mut().write(GreenNodeHead::new(kind, text_len, children_len));
            let children_ptr = node.children_ptr_mut();
            for child_idx in 0..children_len {
                children_ptr.add(child_idx).write(children.next().expect("too few children"));
            }
        }
        debug_assert!(children.next().is_none(), "too many children");
        node
    }

    #[inline]
    pub(crate) fn add_dependency(&self, dep: Arc<GreenTree>) {
        self.dependencies.lock().unwrap().insert(PtrEqArc(dep));
    }
}

impl fmt::Debug for GreenTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreenTree").finish_non_exhaustive()
    }
}
