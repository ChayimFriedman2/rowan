use std::{
    alloc::Layout,
    fmt,
    hash::{Hash, Hasher},
    iter::{self, FusedIterator},
    mem,
    ops::RangeBounds,
    ptr::NonNull,
    slice,
};

use countme::Count;
use triomphe::Arc;

use crate::{
    GreenToken, NodeOrToken, TextRange, TextSize,
    green::{
        GreenElement, GreenElementInTree, SyntaxKind, arena::GreenTree, token::GreenTokenInTree,
    },
};

#[derive(Debug, Clone)]
pub(super) struct GreenNodeHead {
    kind: SyntaxKind,
    text_len: TextSize,
    children_len: u32,
    _c: Count<GreenNode>,
}

// The following impls don't include `children_len`, it will be handled as the slice len.
impl Hash for GreenNodeHead {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Self { kind, text_len, children_len: _, _c: _ } = self;
        kind.hash(state);
        text_len.hash(state);
    }
}

impl PartialEq for GreenNodeHead {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let Self { kind, text_len, children_len: _, _c: _ } = self;
        let Self { kind: other_kind, text_len: other_text_len, children_len: _, _c: _ } = other;
        kind == other_kind && text_len == other_text_len
    }
}

impl Eq for GreenNodeHead {}

impl GreenNodeHead {
    #[inline]
    pub(super) fn layout(children_len: usize) -> Layout {
        Layout::new::<GreenNodeHead>()
            .extend(Layout::array::<GreenChild>(children_len).expect("too big node"))
            .expect("too big node")
            .0
            .pad_to_align()
    }

    #[inline]
    pub(super) fn new(kind: SyntaxKind, text_len: TextSize, children_len: usize) -> Self {
        Self { kind, text_len, children_len: children_len as u32, _c: Count::new() }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub(crate) enum GreenChild {
    Node { rel_offset: TextSize, node: GreenNodeInTree },
    Token { rel_offset: TextSize, token: GreenTokenInTree },
}
#[cfg(target_pointer_width = "64")]
const _: () = assert!(mem::size_of::<GreenChild>() == mem::size_of::<usize>() * 2);

impl GreenChild {
    #[inline]
    pub(crate) fn as_node(&self) -> Option<&GreenNodeInTree> {
        match self {
            GreenChild::Node { rel_offset: _, node } => Some(node),
            GreenChild::Token { .. } => None,
        }
    }

    #[inline]
    pub(crate) fn kind(&self) -> SyntaxKind {
        match self {
            GreenChild::Node { rel_offset: _, node } => node.kind(),
            GreenChild::Token { rel_offset: _, token } => token.kind(),
        }
    }
}

impl fmt::Display for GreenChild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node { rel_offset: _, node } => fmt::Display::fmt(node, f),
            Self::Token { rel_offset: _, token } => fmt::Display::fmt(token, f),
        }
    }
}

#[repr(C)]
pub(super) struct GreenNodeData {
    head: GreenNodeHead,
    children: [GreenChild; 0],
}

#[derive(Clone, Copy)]
pub(crate) struct GreenNodeInTree {
    /// INVARIANT: This points at a valid `GreenNodeHead` then `children_len` `GreenChild`s,
    /// with `#[repr(C)]`.
    pub(super) data: NonNull<GreenNodeData>,
}

// SAFETY: The pointer is valid.
unsafe impl Send for GreenNodeInTree {}
unsafe impl Sync for GreenNodeInTree {}

impl GreenNodeInTree {
    /// Does not require the pointer to be valid.
    #[inline]
    pub(super) fn header_ptr_mut(&self) -> *mut GreenNodeHead {
        // SAFETY: `&raw mut` doesn't require the data to be valid, only allocated.
        unsafe { &raw mut (*self.data.as_ptr()).head }
    }

    #[inline]
    pub(super) fn children_ptr_mut(&self) -> *mut GreenChild {
        // SAFETY: `&raw mut` doesn't require the data to be valid, only allocated.
        unsafe { (&raw mut (*self.data.as_ptr()).children).cast::<GreenChild>() }
    }

    #[inline]
    fn header(&self) -> &GreenNodeHead {
        // SAFETY: `data`'s invariant.
        unsafe { &*self.header_ptr_mut() }
    }

    #[inline]
    pub(crate) fn children(&self) -> &[GreenChild] {
        // SAFETY: `data`'s invariant.
        unsafe {
            slice::from_raw_parts(self.children_ptr_mut(), self.header().children_len as usize)
        }
    }

    #[inline]
    pub(crate) fn kind(&self) -> SyntaxKind {
        self.header().kind
    }

    #[inline]
    pub(crate) fn text_len(&self) -> TextSize {
        self.header().text_len
    }

    #[inline]
    pub(crate) fn as_ptr(self) -> NonNull<()> {
        self.data.cast()
    }

    #[inline]
    pub(crate) fn to_green_node(self, arena: Arc<GreenTree>) -> GreenNode {
        GreenNode { node: self, arena }
    }

    /// # Safety
    ///
    /// You must ensure there is no concurrent allocation.
    #[must_use]
    pub unsafe fn insert_child(
        &self,
        index: usize,
        new_child: GreenElementInTree,
        arena: &GreenTree,
    ) -> GreenNodeInTree {
        // SAFETY: Our precondition.
        unsafe { self.splice_children(index..index, iter::once(new_child), arena) }
    }

    /// # Safety
    ///
    /// You must ensure there is no concurrent allocation.
    #[must_use]
    pub unsafe fn remove_child(&self, index: usize, arena: &GreenTree) -> GreenNodeInTree {
        // SAFETY: Our precondition.
        unsafe { self.splice_children(index..=index, iter::empty(), arena) }
    }

    /// # Safety
    ///
    /// You must ensure there is no concurrent allocation.
    #[must_use]
    pub unsafe fn replace_child(
        &self,
        index: usize,
        new_child: GreenElementInTree,
        arena: &GreenTree,
    ) -> GreenNodeInTree {
        // SAFETY: Our precondition.
        unsafe { self.splice_children(index..=index, iter::once(new_child), arena) }
    }

    /// # Safety
    ///
    /// You must ensure there is no concurrent allocation.
    #[must_use]
    pub unsafe fn splice_children<R, I>(
        &self,
        range: R,
        replace_with: I,
        arena: &GreenTree,
    ) -> GreenNodeInTree
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = GreenElementInTree>,
    {
        let mut children: Vec<_> = self
            .children()
            .iter()
            .map(|child| match *child {
                GreenChild::Node { rel_offset: _, node } => GreenElementInTree::Node(node),
                GreenChild::Token { rel_offset: _, token } => GreenElementInTree::Token(token),
            })
            .collect::<Vec<_>>();
        children.splice(range, replace_with);

        let text_len = children.iter().map(|child| child.text_len()).sum();

        let mut rel_offset = TextSize::new(0);
        let children = children.into_iter().map(|child| match child {
            NodeOrToken::Node(node) => {
                let old_rel_offset = rel_offset;
                rel_offset += node.text_len();
                GreenChild::Node { rel_offset: old_rel_offset, node }
            }
            NodeOrToken::Token(token) => {
                let old_rel_offset = rel_offset;
                rel_offset += token.text_len();
                GreenChild::Token { rel_offset: old_rel_offset, token }
            }
        });

        // SAFETY: Our precondition.
        unsafe { arena.alloc_node_unchecked(self.kind(), text_len, children.len(), children) }
    }
}

impl PartialEq for GreenNodeInTree {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.children() == other.children()
    }
}

impl Eq for GreenNodeInTree {}

impl Hash for GreenNodeInTree {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.header().hash(state);
        self.children().hash(state);
    }
}

/// Internal node in the immutable tree.
/// It has other nodes and tokens as children.
#[derive(Clone)]
pub struct GreenNode {
    pub(super) node: GreenNodeInTree,
    pub(super) arena: Arc<GreenTree>,
}

impl Hash for GreenNode {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
    }
}

impl PartialEq for GreenNode {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl Eq for GreenNode {}

impl fmt::Debug for GreenNodeInTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreenNode")
            .field("kind", &self.header().kind)
            .field("text_len", &self.header().text_len)
            .field("children", &self.children())
            .finish()
    }
}

impl fmt::Debug for GreenNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.node, f)
    }
}

impl fmt::Display for GreenNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.node, f)
    }
}

impl fmt::Display for GreenNodeInTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for child in self.children() {
            write!(f, "{}", child)?;
        }
        Ok(())
    }
}

impl GreenNode {
    #[inline]
    pub(crate) fn into_raw_parts(self) -> (GreenNodeInTree, Arc<GreenTree>) {
        (self.node, self.arena)
    }

    /// Kind of this node.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.node.kind()
    }

    /// Returns the length of the text covered by this node.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        self.node.text_len()
    }

    /// Children of this node.
    #[inline]
    pub fn children(&self) -> Children<'_> {
        Children { raw: self.node.children().iter(), arena: self.arena.clone() }
    }
}

impl GreenChild {
    #[inline]
    pub(crate) fn as_green_element(&self, arena: Arc<GreenTree>) -> GreenElement {
        match self {
            GreenChild::Node { node, .. } => NodeOrToken::Node(GreenNode { node: *node, arena }),
            GreenChild::Token { token, .. } => {
                NodeOrToken::Token(GreenToken { token: *token, _arena: arena })
            }
        }
    }

    #[inline]
    pub(crate) fn rel_offset(&self) -> TextSize {
        match self {
            GreenChild::Node { rel_offset, .. } | GreenChild::Token { rel_offset, .. } => {
                *rel_offset
            }
        }
    }
    #[inline]
    pub(crate) fn rel_range(&self) -> TextRange {
        match *self {
            GreenChild::Node { rel_offset, node } => TextRange::at(rel_offset, node.text_len()),
            GreenChild::Token { rel_offset, token } => TextRange::at(rel_offset, token.text_len()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Children<'a> {
    pub(crate) raw: slice::Iter<'a, GreenChild>,
    arena: Arc<GreenTree>,
}

// NB: forward everything stable that iter::Slice specializes as of Rust 1.39.0
impl ExactSizeIterator for Children<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.raw.len()
    }
}

impl<'a> Iterator for Children<'a> {
    type Item = GreenElement;

    #[inline]
    fn next(&mut self) -> Option<GreenElement> {
        self.raw.next().map(|child| child.as_green_element(self.arena.clone()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint()
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.raw.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth(n).map(|child| child.as_green_element(self.arena.clone()))
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, mut f: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        for x in self {
            accum = f(accum, x);
        }
        accum
    }
}

impl DoubleEndedIterator for Children<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.raw.next_back().map(|child| child.as_green_element(self.arena.clone()))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.raw.nth_back(n).map(|child| child.as_green_element(self.arena.clone()))
    }

    #[inline]
    fn rfold<Acc, Fold>(mut self, init: Acc, mut f: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        while let Some(x) = self.next_back() {
            accum = f(accum, x);
        }
        accum
    }
}

impl FusedIterator for Children<'_> {}
