//! Implementation of the cursors -- API for convenient access to syntax trees.
//!
//! Functional programmers will recognize that this module implements a zipper
//! for a purely functional (green) tree.
//!
//! A cursor node (`SyntaxNode`) points to a `GreenNode` and a parent
//! `SyntaxNode`. This allows cursor to provide iteration over both ancestors
//! and descendants, as well as a cheep access to absolute offset of the node in
//! file.
//!
//! By default `SyntaxNode`s are immutable, but you can get a mutable copy of
//! the tree by calling `clone_for_update`. Mutation is based on interior
//! mutability and doesn't need `&mut`. You can have two `SyntaxNode`s pointing
//! at different parts of the same tree; mutations via the first node will be
//! reflected in the other.

// Implementation notes:
//
// The implementation is utterly and horribly unsafe. This whole module is an
// unsafety boundary. It is believed that the API here is, in principle, sound,
// but the implementation might have bugs.
//
// The core type is `NodeData` -- a heap-allocated reference counted object,
// which points to a green node or a green token, and to the parent `NodeData`.
// Publicly-exposed `SyntaxNode` and `SyntaxToken` own a reference to
// `NodeData`.
//
// `NodeData`s are transient, and are created and destroyed during tree
// traversals. In general, only currently referenced nodes and their ancestors
// are alive at any given moment.
//
// More specifically, `NodeData`'s ref count is equal to the number of
// outstanding `SyntaxNode` and `SyntaxToken` plus the number of children with
// non-zero ref counts. For example, if the user has only a single `SyntaxNode`
// pointing somewhere in the middle of the tree, then all `NodeData` on the path
// from that point towards the root have ref count equal to one.
//
// `NodeData` which doesn't have a parent (is a root) owns the corresponding
// green node or token, and is responsible for freeing it.
//
// That's mostly it for the immutable subset of the API. Mutation is fun though,
// you'll like it!
//
// Mutability is a run-time property of a tree of `NodeData`. The whole tree is
// either mutable or immutable. `clone_for_update` clones the whole tree of
// `NodeData`s, making it mutable (note that the green tree is re-used).
//
// If the tree is mutable, then all live `NodeData` are additionally liked to
// each other via intrusive liked lists. Specifically, there are two pointers to
// siblings, as well as a pointer to the first child. Note that only live nodes
// are considered. If the user only has `SyntaxNode`s for  the first and last
// children of some particular node, then their `NodeData` will point at each
// other.
//
// The links are used to propagate mutations across the tree. Specifically, each
// `NodeData` remembers it's index in parent. When the node is detached from or
// attached to the tree, we need to adjust the indices of all subsequent
// siblings. That's what makes the `for c in node.children() { c.detach() }`
// pattern work despite the apparent iterator invalidation.
//
// This code is encapsulated into the sorted linked list (`sll`) module.
//
// The actual mutation consist of functionally "mutating" (creating a
// structurally shared copy) the green node, and then re-spinning the tree. This
// is a delicate process: `NodeData` point directly to the green nodes, so we
// must make sure that those nodes don't move. Additionally, during mutation a
// node might become or might stop being a root, so we must take care to not
// double free / leak its green node.
//
// Because we can change green nodes using only shared references, handing out
// references into green nodes in the public API would be unsound. We don't do
// that, but we do use such references internally a lot. Additionally, for
// tokens the underlying green token actually is immutable, so we can, and do
// return `&str`.
//
// Invariants [must not leak outside of the module]:
//    - Mutability is the property of the whole tree. Intermixing elements that
//      differ in mutability is not allowed.
//    - Mutability property is persistent.
//    - References to the green elements' data are not exposed into public API
//      when the tree is mutable.
//    - TBD

mod arena;

use std::{
    cell::Cell,
    fmt,
    hash::{Hash, Hasher},
    iter,
    ops::Range,
    ptr,
    rc::Rc,
};

use countme::Count;

use crate::{
    Direction, GreenNode, GreenToken, NodeOrToken, SyntaxText, TextRange, TextSize, TokenAtOffset,
    WalkEvent,
    cursor::arena::RedTree,
    green::{GreenChild, GreenElementInTree, GreenNodeInTree, GreenTokenInTree, SyntaxKind},
    sll,
};

#[derive(Debug)]
enum Green {
    Node { ptr: Cell<GreenNodeInTree> },
    Token { ptr: GreenTokenInTree },
}

impl Green {
    #[inline]
    fn as_node(&self) -> Option<GreenNodeInTree> {
        match self {
            Green::Node { ptr } => Some(ptr.get()),
            Green::Token { .. } => None,
        }
    }

    #[inline]
    fn as_token(&self) -> Option<&GreenTokenInTree> {
        match self {
            Green::Token { ptr } => Some(ptr),
            Green::Node { .. } => None,
        }
    }
}

#[derive(Debug)]
struct _SyntaxElement;

#[derive(Debug)]
struct NodeData {
    _c: Count<_SyntaxElement>,

    parent: Cell<Option<ptr::NonNull<NodeData>>>,
    index: Cell<u32>,
    green: Green,

    /// Absolute offset for immutable nodes, unused for mutable nodes.
    offset: TextSize,
    // The following links only have meaning when `mutable` is true.
    first: Cell<*const NodeData>,
    /// Invariant: never null if mutable.
    next: Cell<*const NodeData>,
    /// Invariant: never null if mutable.
    prev: Cell<*const NodeData>,
}

unsafe impl sll::Elem for NodeData {
    fn prev(&self) -> &Cell<*const Self> {
        &self.prev
    }
    fn next(&self) -> &Cell<*const Self> {
        &self.next
    }
    fn key(&self) -> &Cell<u32> {
        &self.index
    }
}

pub type SyntaxElement = NodeOrToken<SyntaxNode, SyntaxToken>;

#[derive(Clone)]
pub struct SyntaxNode {
    ptr: ptr::NonNull<NodeData>,
    arena: Rc<RedTree>,
}

#[derive(Clone)]
pub struct SyntaxToken {
    ptr: ptr::NonNull<NodeData>,
    arena: Rc<RedTree>,
}

type NodeKey = (ptr::NonNull<()>, TextSize);

impl NodeData {
    #[inline]
    fn key(&self, use_slow_offset: bool) -> NodeKey {
        let ptr = match &self.green {
            Green::Node { ptr } => ptr.get().as_ptr(),
            Green::Token { ptr } => ptr.as_ptr(),
        };
        (ptr, self.offset(use_slow_offset))
    }

    #[inline]
    fn parent_node(&self, arena: Rc<RedTree>) -> Option<SyntaxNode> {
        let parent = self.parent.get()?;
        debug_assert!(matches!(unsafe { &parent.as_ref().green }, Green::Node { .. }));
        Some(SyntaxNode { ptr: parent, arena })
    }

    #[inline]
    fn green(&self) -> GreenElementInTree {
        match &self.green {
            Green::Node { ptr } => GreenElementInTree::Node(ptr.get()),
            Green::Token { ptr } => GreenElementInTree::Token(*ptr),
        }
    }

    #[inline]
    fn green_siblings(&self) -> &[GreenChild] {
        match self.parent.get().map(|it| unsafe { &it.as_ref().green }) {
            Some(Green::Node { ptr }) => {
                let node = ptr.get();
                // Lifetime-extend it.
                // SAFETY: The parent will live as long as `self` will (because we never deallocate nodes).
                // And we tie the reference to `self`.
                unsafe { &*ptr::from_ref(node.children()) }
            }
            Some(Green::Token { .. }) => {
                debug_assert!(false);
                &[]
            }
            None => &[],
        }
    }

    #[inline]
    fn index(&self) -> u32 {
        self.index.get()
    }

    #[inline]
    fn offset(&self, use_slow_offset: bool) -> TextSize {
        if use_slow_offset { self.offset_mut() } else { self.offset }
    }

    #[cold]
    fn offset_mut(&self) -> TextSize {
        let mut res = TextSize::from(0);

        let mut node = self;
        while let Some(parent) = node.parent.get() {
            let parent = unsafe { parent.as_ref() };
            let green = parent.green.as_node().unwrap();
            res += green.children()[node.index() as usize].rel_offset();
            node = parent;
        }

        res
    }

    #[inline]
    fn text_range(&self, arena: &RedTree) -> TextRange {
        let offset = self.offset(arena.mutable());
        let len = self.green().text_len();
        TextRange::at(offset, len)
    }

    #[inline]
    fn kind(&self) -> SyntaxKind {
        self.green().kind()
    }

    #[inline]
    fn next_sibling(&self, arena: &Rc<RedTree>) -> Option<SyntaxNode> {
        let siblings = self.green_siblings().iter().enumerate();
        let index = self.index() as usize;

        let parent = self.parent.get()?;
        siblings.skip(index + 1).find_map(|(index, child)| {
            child.as_node().map(|green| {
                let parent_offset = unsafe { parent.as_ref() }.offset(arena.mutable());
                let offset = parent_offset + child.rel_offset();
                // SAFETY: The green node came from the same syntax tree as us.
                unsafe {
                    SyntaxNode::new_child(arena.clone(), *green, parent, index as u32, offset)
                }
            })
        })
    }

    #[inline]
    fn next_sibling_by_kind(
        &self,
        mut matcher: impl FnMut(SyntaxKind) -> bool,
        arena: &Rc<RedTree>,
    ) -> Option<SyntaxNode> {
        let siblings = self.green_siblings().iter().enumerate();
        let index = self.index() as usize;

        let parent = self.parent.get()?;
        siblings.skip(index + 1).find_map(|(index, child)| {
            let &GreenChild::Node { rel_offset, node } = child else { return None };
            if !matcher(child.kind()) {
                return None;
            }

            let parent_offset = unsafe { parent.as_ref() }.offset(arena.mutable());
            let offset = parent_offset + rel_offset;
            // SAFETY: The green node came from the same syntax tree as us.
            Some(unsafe {
                SyntaxNode::new_child(arena.clone(), node, parent, index as u32, offset)
            })
        })
    }

    #[inline]
    fn prev_sibling(&self, arena: &Rc<RedTree>) -> Option<SyntaxNode> {
        let index = self.index() as usize;
        let mut siblings = self.green_siblings()[..index].iter().enumerate().rev();

        let parent = self.parent.get()?;
        siblings.find_map(|(index, child)| {
            child.as_node().map(|green| {
                let parent_offset = unsafe { parent.as_ref() }.offset(arena.mutable());
                let offset = parent_offset + child.rel_offset();
                // SAFETY: The green node came from the same syntax tree as us.
                unsafe {
                    SyntaxNode::new_child(arena.clone(), *green, parent, index as u32, offset)
                }
            })
        })
    }

    #[inline]
    fn next_sibling_or_token(&self, arena: &Rc<RedTree>) -> Option<SyntaxElement> {
        let siblings = self.green_siblings();
        let index = self.index() as usize + 1;

        siblings.get(index).and_then(|child| {
            let parent = self.parent.get()?;
            let parent_offset = unsafe { parent.as_ref() }.offset(arena.mutable());
            let offset = parent_offset + child.rel_offset();
            // SAFETY: The green node came from the same syntax tree as us.
            Some(unsafe { SyntaxElement::new(arena.clone(), *child, parent, index as u32, offset) })
        })
    }

    #[inline]
    fn next_sibling_or_token_by_kind(
        &self,
        mut matcher: impl FnMut(SyntaxKind) -> bool,
        arena: &Rc<RedTree>,
    ) -> Option<SyntaxElement> {
        let siblings = self.green_siblings().iter().enumerate();
        let index = self.index() as usize;

        let parent = self.parent.get()?;
        siblings.skip(index + 1).find_map(|(index, child)| {
            if !matcher(child.kind()) {
                return None;
            }
            let parent_offset = unsafe { parent.as_ref() }.offset(arena.mutable());
            let offset = parent_offset + child.rel_offset();
            // SAFETY: The green node came from the same syntax tree as us.
            Some(unsafe { SyntaxElement::new(arena.clone(), *child, parent, index as u32, offset) })
        })
    }

    #[inline]
    fn prev_sibling_or_token(&self, arena: &Rc<RedTree>) -> Option<SyntaxElement> {
        let siblings = self.green_siblings();
        let index = self.index().checked_sub(1)? as usize;

        siblings.get(index).and_then(|child| {
            let parent = self.parent.get()?;
            let parent_offset = unsafe { parent.as_ref() }.offset(arena.mutable());
            let offset = parent_offset + child.rel_offset();
            // SAFETY: The green node came from the same syntax tree as us.
            Some(unsafe { SyntaxElement::new(arena.clone(), *child, parent, index as u32, offset) })
        })
    }
}

impl SyntaxNode {
    #[inline]
    pub fn new_root(green: GreenNode) -> SyntaxNode {
        RedTree::new(green)
    }

    #[inline]
    pub fn new_root_mut(green: GreenNode) -> SyntaxNode {
        RedTree::new_mut(green)
    }

    /// # Safety
    ///
    /// You need to make sure `green` comes from `self.arena.green` or one of the secondary green trees.
    #[inline]
    unsafe fn new_child(
        arena: Rc<RedTree>,
        green: GreenNodeInTree,
        parent: ptr::NonNull<NodeData>,
        index: u32,
        offset: TextSize,
    ) -> SyntaxNode {
        let green = Green::Node { ptr: Cell::new(green) };
        // SAFETY: Our precondition.
        let node = unsafe { arena.alloc_node(Some(parent), index, offset, green) };
        SyntaxNode { ptr: node, arena }
    }

    pub fn is_mutable(&self) -> bool {
        self.arena.mutable()
    }

    pub fn clone_for_update(&self) -> SyntaxNode {
        assert!(!self.arena.mutable());
        match self.parent() {
            Some(parent) => {
                let parent = parent.clone_for_update();
                unsafe {
                    SyntaxNode::new_child(
                        parent.arena.clone(),
                        self.green_in_tree(),
                        parent.ptr,
                        self.data().index(),
                        self.offset(),
                    )
                }
            }
            None => SyntaxNode::new_root_mut(self.green()),
        }
    }

    pub fn clone_subtree(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green())
    }

    #[inline]
    fn data(&self) -> &NodeData {
        unsafe { self.ptr.as_ref() }
    }

    pub fn replace_with(&self, replacement: GreenNode) -> GreenNode {
        self.arena.replace_with(self.clone().into(), replacement.into())
    }

    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.data().kind()
    }

    #[inline]
    fn offset(&self) -> TextSize {
        self.data().offset(self.arena.mutable())
    }

    #[inline]
    pub fn text_range(&self) -> TextRange {
        self.data().text_range(&self.arena)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.data().index() as usize
    }

    #[inline]
    pub fn text(&self) -> SyntaxText {
        SyntaxText::new(self.clone())
    }

    #[inline]
    pub fn green(&self) -> GreenNode {
        self.arena.green_node(self)
    }

    #[inline]
    fn green_in_tree(&self) -> GreenNodeInTree {
        self.data().green.as_node().unwrap()
    }

    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        self.data().parent_node(self.arena.clone())
    }

    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> + use<> {
        iter::successors(Some(self.clone()), SyntaxNode::parent)
    }

    #[inline]
    pub fn children(&self) -> SyntaxNodeChildren {
        SyntaxNodeChildren::new(self.clone())
    }

    #[inline]
    pub fn children_with_tokens(&self) -> SyntaxElementChildren {
        SyntaxElementChildren::new(self.clone())
    }

    #[inline]
    pub fn first_child(&self) -> Option<SyntaxNode> {
        self.green_in_tree().children().iter().enumerate().find_map(|(index, child)| {
            child.as_node().map(|green| unsafe {
                SyntaxNode::new_child(
                    self.arena.clone(),
                    *green,
                    self.ptr,
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            })
        })
    }

    #[inline]
    pub fn first_child_by_kind(
        &self,
        mut matcher: impl FnMut(SyntaxKind) -> bool,
    ) -> Option<SyntaxNode> {
        self.green_in_tree().children().iter().enumerate().find_map(|(index, child)| {
            let child_node = *child.as_node()?;
            if !matcher(child_node.kind()) {
                return None;
            }
            unsafe {
                Some(SyntaxNode::new_child(
                    self.arena.clone(),
                    child_node,
                    self.ptr,
                    index as u32,
                    self.offset() + child.rel_offset(),
                ))
            }
        })
    }

    #[inline]
    pub fn last_child(&self) -> Option<SyntaxNode> {
        self.green_in_tree().children().iter().enumerate().rev().find_map(|(index, child)| {
            child.as_node().map(|green| unsafe {
                SyntaxNode::new_child(
                    self.arena.clone(),
                    *green,
                    self.ptr,
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            })
        })
    }

    #[inline]
    pub fn first_child_or_token(&self) -> Option<SyntaxElement> {
        self.green_in_tree().children().first().map(|child| unsafe {
            SyntaxElement::new(
                self.arena.clone(),
                *child,
                self.ptr,
                0,
                self.offset() + child.rel_offset(),
            )
        })
    }

    #[inline]
    pub fn first_child_or_token_by_kind(
        &self,
        mut matcher: impl FnMut(SyntaxKind) -> bool,
    ) -> Option<SyntaxElement> {
        self.green_in_tree().children().iter().enumerate().find_map(|(index, child)| {
            if !matcher(child.kind()) {
                return None;
            }
            Some(unsafe {
                SyntaxElement::new(
                    self.arena.clone(),
                    *child,
                    self.ptr,
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            })
        })
    }

    pub fn last_child_or_token(&self) -> Option<SyntaxElement> {
        self.green_in_tree().children().iter().enumerate().next_back().map(
            |(index, child)| unsafe {
                SyntaxElement::new(
                    self.arena.clone(),
                    *child,
                    self.ptr,
                    index as u32,
                    self.offset() + child.rel_offset(),
                )
            },
        )
    }

    #[inline]
    pub fn to_next_sibling(self) -> Option<SyntaxNode> {
        // FIXME: This can avoid the refcount bump & drop on the arena.
        self.next_sibling()
    }

    #[inline]
    pub fn next_sibling(&self) -> Option<SyntaxNode> {
        self.data().next_sibling(&self.arena)
    }

    #[inline]
    pub fn next_sibling_by_kind(
        &self,
        matcher: impl FnMut(SyntaxKind) -> bool,
    ) -> Option<SyntaxNode> {
        self.data().next_sibling_by_kind(matcher, &self.arena)
    }

    #[inline]
    pub fn prev_sibling(&self) -> Option<SyntaxNode> {
        self.data().prev_sibling(&self.arena)
    }

    #[inline]
    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().next_sibling_or_token(&self.arena)
    }

    #[inline]
    pub fn next_sibling_or_token_by_kind(
        &self,
        matcher: impl FnMut(SyntaxKind) -> bool,
    ) -> Option<SyntaxElement> {
        self.data().next_sibling_or_token_by_kind(matcher, &self.arena)
    }

    #[inline]
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().prev_sibling_or_token(&self.arena)
    }

    #[inline]
    pub fn first_token(&self) -> Option<SyntaxToken> {
        self.first_child_or_token()?.first_token()
    }
    #[inline]
    pub fn last_token(&self) -> Option<SyntaxToken> {
        self.last_child_or_token()?.last_token()
    }

    #[inline]
    pub fn siblings(&self, direction: Direction) -> impl Iterator<Item = SyntaxNode> + use<> {
        iter::successors(Some(self.clone()), move |node| match direction {
            Direction::Next => node.next_sibling(),
            Direction::Prev => node.prev_sibling(),
        })
    }

    #[inline]
    pub fn siblings_with_tokens(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = SyntaxElement> + use<> {
        let me: SyntaxElement = self.clone().into();
        iter::successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }

    #[inline]
    pub fn descendants(&self) -> impl Iterator<Item = SyntaxNode> + use<> {
        self.preorder().filter_map(|event| match event {
            WalkEvent::Enter(node) => Some(node),
            WalkEvent::Leave(_) => None,
        })
    }

    #[inline]
    pub fn descendants_with_tokens(&self) -> impl Iterator<Item = SyntaxElement> + use<> {
        self.preorder_with_tokens().filter_map(|event| match event {
            WalkEvent::Enter(it) => Some(it),
            WalkEvent::Leave(_) => None,
        })
    }

    #[inline]
    pub fn preorder(&self) -> Preorder {
        Preorder::new(self.clone())
    }

    #[inline]
    pub fn preorder_with_tokens(&self) -> PreorderWithTokens {
        PreorderWithTokens::new(self.clone())
    }

    pub fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        // TODO: this could be faster if we first drill-down to node, and only
        // then switch to token search. We should also replace explicit
        // recursion with a loop.
        let range = self.text_range();
        assert!(
            range.start() <= offset && offset <= range.end(),
            "Bad offset: range {:?} offset {:?}",
            range,
            offset
        );
        if range.is_empty() {
            return TokenAtOffset::None;
        }

        let mut children = self.children_with_tokens().filter(|child| {
            let child_range = child.text_range();
            !child_range.is_empty()
                && (child_range.start() <= offset && offset <= child_range.end())
        });

        let left = children.next().unwrap();
        let right = children.next();
        assert!(children.next().is_none());

        if let Some(right) = right {
            match (left.token_at_offset(offset), right.token_at_offset(offset)) {
                (TokenAtOffset::Single(left), TokenAtOffset::Single(right)) => {
                    TokenAtOffset::Between(left, right)
                }
                _ => unreachable!(),
            }
        } else {
            left.token_at_offset(offset)
        }
    }

    pub fn covering_element(&self, range: TextRange) -> SyntaxElement {
        let mut res: SyntaxElement = self.clone().into();
        loop {
            assert!(
                res.text_range().contains_range(range),
                "Bad range: node range {:?}, range {:?}",
                res.text_range(),
                range,
            );
            res = match &res {
                NodeOrToken::Token(_) => return res,
                NodeOrToken::Node(node) => match node.child_or_token_at_range(range) {
                    Some(it) => it,
                    None => return res,
                },
            };
        }
    }

    pub fn child_or_token_at_range(&self, range: TextRange) -> Option<SyntaxElement> {
        let rel_range = range - self.offset();

        let green = self.green_in_tree();
        let index = green
            .children()
            .binary_search_by(|it| {
                let child_range = it.rel_range();
                TextRange::ordering(child_range, rel_range)
            })
            // XXX: this handles empty ranges
            .unwrap_or_else(|it| it.saturating_sub(1));
        let child =
            green.children().get(index).filter(|it| it.rel_range().contains_range(rel_range))?;
        let rel_offset = child.rel_offset();

        Some(unsafe {
            SyntaxElement::new(
                self.arena.clone(),
                *child,
                self.ptr,
                index as u32,
                self.offset() + rel_offset,
            )
        })
    }

    pub fn splice_children<I: IntoIterator<Item = SyntaxElement>>(
        &self,
        to_delete: Range<usize>,
        to_insert: I,
    ) {
        assert!(self.arena.mutable(), "immutable tree: {}", self);
        for (i, child) in self.children_with_tokens().enumerate() {
            if to_delete.contains(&i) {
                child.detach();
            }
        }
        let mut index = to_delete.start;
        for child in to_insert {
            self.attach_child(index, child);
            index += 1;
        }
    }

    pub fn detach(&self) {
        assert!(self.arena.mutable(), "immutable tree: {}", self);
        self.arena.detach(self.data())
    }

    fn attach_child(&self, index: usize, child: SyntaxElement) {
        assert!(self.arena.mutable(), "immutable tree: {}", self);
        child.detach();
        self.arena.attach_node_child(self.ptr, index, child);
    }
}

impl SyntaxToken {
    /// # Safety
    ///
    /// You need to make sure `green` comes from `self.arena.green` or one of the secondary green trees.
    unsafe fn new(
        arena: Rc<RedTree>,
        green: GreenTokenInTree,
        parent: ptr::NonNull<NodeData>,
        index: u32,
        offset: TextSize,
    ) -> SyntaxToken {
        let green = Green::Token { ptr: green.into() };
        // SAFETY: Our precondition.
        SyntaxToken { ptr: unsafe { arena.alloc_node(Some(parent), index, offset, green) }, arena }
    }

    #[inline]
    fn data(&self) -> &NodeData {
        unsafe { self.ptr.as_ref() }
    }

    pub fn replace_with(&self, replacement: GreenToken) -> GreenNode {
        self.arena.replace_with(self.clone().into(), replacement.into())
    }

    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.data().kind()
    }

    #[inline]
    pub fn text_range(&self) -> TextRange {
        self.data().text_range(&self.arena)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.data().index() as usize
    }

    #[inline]
    pub fn text(&self) -> &str {
        match &self.data().green {
            Green::Token { ptr } => ptr.text(),
            Green::Node { .. } => {
                panic!(
                    "corrupted tree: a node thinks it is a token: {:?}",
                    self.data().green.as_node().unwrap().to_string()
                )
            }
        }
    }

    #[inline]
    pub fn green(&self) -> GreenToken {
        self.arena.green_token(self)
    }

    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        self.data().parent_node(self.arena.clone())
    }

    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> + use<> {
        std::iter::successors(self.parent(), SyntaxNode::parent)
    }

    #[inline]
    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().next_sibling_or_token(&self.arena)
    }

    #[inline]
    pub fn next_sibling_or_token_by_kind(
        &self,
        matcher: impl FnMut(SyntaxKind) -> bool,
    ) -> Option<SyntaxElement> {
        self.data().next_sibling_or_token_by_kind(matcher, &self.arena)
    }

    #[inline]
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        self.data().prev_sibling_or_token(&self.arena)
    }

    #[inline]
    pub fn siblings_with_tokens(
        &self,
        direction: Direction,
    ) -> impl Iterator<Item = SyntaxElement> + use<> {
        let me: SyntaxElement = self.clone().into();
        iter::successors(Some(me), move |el| match direction {
            Direction::Next => el.next_sibling_or_token(),
            Direction::Prev => el.prev_sibling_or_token(),
        })
    }

    #[inline]
    pub fn next_token(&self) -> Option<SyntaxToken> {
        match self.next_sibling_or_token() {
            Some(element) => element.first_token(),
            None => self
                .ancestors()
                .find_map(|it| it.next_sibling_or_token())
                .and_then(|element| element.first_token()),
        }
    }
    #[inline]
    pub fn prev_token(&self) -> Option<SyntaxToken> {
        match self.prev_sibling_or_token() {
            Some(element) => element.last_token(),
            None => self
                .ancestors()
                .find_map(|it| it.prev_sibling_or_token())
                .and_then(|element| element.last_token()),
        }
    }

    pub fn detach(&self) {
        assert!(self.arena.mutable(), "immutable tree: {}", self);
        self.arena.detach(self.data());
    }
}

impl SyntaxElement {
    /// # Safety
    ///
    /// You need to make sure `green` comes from `self.arena.green` or one of the secondary green trees.
    #[inline]
    unsafe fn new(
        arena: Rc<RedTree>,
        green: GreenChild,
        parent: ptr::NonNull<NodeData>,
        index: u32,
        offset: TextSize,
    ) -> SyntaxElement {
        match green {
            // SAFETY: Our precondition.
            GreenChild::Node { rel_offset: _, node } => unsafe {
                SyntaxNode::new_child(arena, node, parent, index, offset).into()
            },
            // SAFETY: Our precondition.
            GreenChild::Token { rel_offset: _, token } => unsafe {
                SyntaxToken::new(arena, token, parent, index, offset).into()
            },
        }
    }

    #[inline]
    fn data(&self) -> &NodeData {
        unsafe { self.data_ptr().as_ref() }
    }

    #[inline]
    fn data_ptr(&self) -> ptr::NonNull<NodeData> {
        match self {
            NodeOrToken::Node(it) => it.ptr,
            NodeOrToken::Token(it) => it.ptr,
        }
    }

    #[inline]
    pub fn text_range(&self) -> TextRange {
        match self {
            NodeOrToken::Node(it) => it.text_range(),
            NodeOrToken::Token(it) => it.text_range(),
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        match self {
            NodeOrToken::Node(it) => it.index(),
            NodeOrToken::Token(it) => it.index(),
        }
    }

    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }

    #[inline]
    pub fn parent(&self) -> Option<SyntaxNode> {
        match self {
            NodeOrToken::Node(it) => it.parent(),
            NodeOrToken::Token(it) => it.parent(),
        }
    }

    #[inline]
    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> + use<> {
        let first = match self {
            NodeOrToken::Node(it) => Some(it.clone()),
            NodeOrToken::Token(it) => it.parent(),
        };
        iter::successors(first, SyntaxNode::parent)
    }

    #[inline]
    pub fn first_token(&self) -> Option<SyntaxToken> {
        match self {
            NodeOrToken::Node(it) => it.first_token(),
            NodeOrToken::Token(it) => Some(it.clone()),
        }
    }
    #[inline]
    pub fn last_token(&self) -> Option<SyntaxToken> {
        match self {
            NodeOrToken::Node(it) => it.last_token(),
            NodeOrToken::Token(it) => Some(it.clone()),
        }
    }

    #[inline]
    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.next_sibling_or_token(),
            NodeOrToken::Token(it) => it.next_sibling_or_token(),
        }
    }

    #[inline]
    pub fn to_next_sibling_or_token(self) -> Option<SyntaxElement> {
        // FIXME: This can avoid the refcount bump & drop.
        self.next_sibling_or_token()
    }

    #[inline]
    pub fn next_sibling_or_token_by_kind(
        &self,
        matcher: impl FnMut(SyntaxKind) -> bool,
    ) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.next_sibling_or_token_by_kind(matcher),
            NodeOrToken::Token(it) => it.next_sibling_or_token_by_kind(matcher),
        }
    }

    #[inline]
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        match self {
            NodeOrToken::Node(it) => it.prev_sibling_or_token(),
            NodeOrToken::Token(it) => it.prev_sibling_or_token(),
        }
    }

    fn token_at_offset(&self, offset: TextSize) -> TokenAtOffset<SyntaxToken> {
        assert!(self.text_range().start() <= offset && offset <= self.text_range().end());
        match self {
            NodeOrToken::Token(token) => TokenAtOffset::Single(token.clone()),
            NodeOrToken::Node(node) => node.token_at_offset(offset),
        }
    }

    pub fn detach(&self) {
        match self {
            NodeOrToken::Node(it) => it.detach(),
            NodeOrToken::Token(it) => it.detach(),
        }
    }
}

// region: impls

// Identity semantics for hash & eq
impl PartialEq for SyntaxNode {
    #[inline]
    fn eq(&self, other: &SyntaxNode) -> bool {
        self.data().key(self.arena.mutable()) == other.data().key(self.arena.mutable())
    }
}

impl Eq for SyntaxNode {}

impl Hash for SyntaxNode {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().key(self.arena.mutable()).hash(state);
    }
}

impl fmt::Debug for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyntaxNode")
            .field("kind", &self.kind())
            .field("text_range", &self.text_range())
            .finish()
    }
}

impl fmt::Display for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.preorder_with_tokens()
            .filter_map(|event| match event {
                WalkEvent::Enter(NodeOrToken::Token(token)) => Some(token),
                _ => None,
            })
            .try_for_each(|it| fmt::Display::fmt(&it, f))
    }
}

// Identity semantics for hash & eq
impl PartialEq for SyntaxToken {
    #[inline]
    fn eq(&self, other: &SyntaxToken) -> bool {
        self.data().key(self.arena.mutable()) == other.data().key(self.arena.mutable())
    }
}

impl Eq for SyntaxToken {}

impl Hash for SyntaxToken {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data().key(self.arena.mutable()).hash(state);
    }
}

impl fmt::Debug for SyntaxToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyntaxToken").field("text", &self.text()).finish()
    }
}

impl fmt::Display for SyntaxToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.text(), f)
    }
}

impl From<SyntaxNode> for SyntaxElement {
    #[inline]
    fn from(node: SyntaxNode) -> SyntaxElement {
        NodeOrToken::Node(node)
    }
}

impl From<SyntaxToken> for SyntaxElement {
    #[inline]
    fn from(token: SyntaxToken) -> SyntaxElement {
        NodeOrToken::Token(token)
    }
}

// endregion

// region: iterators

#[derive(Clone, Debug)]
pub struct SyntaxNodeChildren {
    parent: SyntaxNode,
    next: Option<SyntaxNode>,
    next_initialized: bool,
}

impl SyntaxNodeChildren {
    fn new(parent: SyntaxNode) -> SyntaxNodeChildren {
        SyntaxNodeChildren { parent, next: None, next_initialized: false }
    }

    pub fn by_kind<F: Fn(SyntaxKind) -> bool>(self, matcher: F) -> SyntaxNodeChildrenByKind<F> {
        if !self.next_initialized {
            SyntaxNodeChildrenByKind { next: self.parent.first_child_by_kind(&matcher), matcher }
        } else {
            SyntaxNodeChildrenByKind {
                next: self.next.and_then(|node| {
                    if matcher(node.kind()) {
                        Some(node)
                    } else {
                        node.next_sibling_by_kind(&matcher)
                    }
                }),
                matcher,
            }
        }
    }
}

impl Iterator for SyntaxNodeChildren {
    type Item = SyntaxNode;
    fn next(&mut self) -> Option<SyntaxNode> {
        if !self.next_initialized {
            self.next = self.parent.first_child();
            self.next_initialized = true;
        } else {
            self.next = self.next.take().and_then(|next| next.to_next_sibling());
        }

        self.next.clone()
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxNodeChildrenByKind<F: Fn(SyntaxKind) -> bool> {
    next: Option<SyntaxNode>,
    matcher: F,
}

impl<F: Fn(SyntaxKind) -> bool> Iterator for SyntaxNodeChildrenByKind<F> {
    type Item = SyntaxNode;
    fn next(&mut self) -> Option<SyntaxNode> {
        self.next.take().inspect(|next| {
            self.next = next.next_sibling_by_kind(&self.matcher);
        })
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxElementChildren {
    parent: SyntaxNode,
    next: Option<SyntaxElement>,
    next_initialized: bool,
}

impl SyntaxElementChildren {
    fn new(parent: SyntaxNode) -> SyntaxElementChildren {
        SyntaxElementChildren { parent, next: None, next_initialized: false }
    }

    pub fn by_kind<F: Fn(SyntaxKind) -> bool>(self, matcher: F) -> SyntaxElementChildrenByKind<F> {
        if !self.next_initialized {
            SyntaxElementChildrenByKind {
                next: self.parent.first_child_or_token_by_kind(&matcher),
                matcher,
            }
        } else {
            SyntaxElementChildrenByKind {
                next: self.next.and_then(|node| {
                    if matcher(node.kind()) {
                        Some(node)
                    } else {
                        node.next_sibling_or_token_by_kind(&matcher)
                    }
                }),
                matcher,
            }
        }
    }
}

impl Iterator for SyntaxElementChildren {
    type Item = SyntaxElement;
    fn next(&mut self) -> Option<SyntaxElement> {
        if !self.next_initialized {
            self.next = self.parent.first_child_or_token();
            self.next_initialized = true;
        } else {
            self.next = self.next.take().and_then(|next| next.to_next_sibling_or_token());
        }

        self.next.clone()
    }
}

#[derive(Clone, Debug)]
pub struct SyntaxElementChildrenByKind<F: Fn(SyntaxKind) -> bool> {
    next: Option<SyntaxElement>,
    matcher: F,
}

impl<F: Fn(SyntaxKind) -> bool> Iterator for SyntaxElementChildrenByKind<F> {
    type Item = SyntaxElement;
    fn next(&mut self) -> Option<SyntaxElement> {
        self.next.take().inspect(|next| {
            self.next = next.next_sibling_or_token_by_kind(&self.matcher);
        })
    }
}

pub struct Preorder {
    start: SyntaxNode,
    next: Option<WalkEvent<SyntaxNode>>,
    skip_subtree: bool,
}

impl Preorder {
    fn new(start: SyntaxNode) -> Preorder {
        let next = Some(WalkEvent::Enter(start.clone()));
        Preorder { start, next, skip_subtree: false }
    }

    pub fn skip_subtree(&mut self) {
        self.skip_subtree = true;
    }

    #[cold]
    fn do_skip(&mut self) {
        self.next = self.next.take().map(|next| match next {
            WalkEvent::Enter(first_child) => WalkEvent::Leave(first_child.parent().unwrap()),
            WalkEvent::Leave(parent) => WalkEvent::Leave(parent),
        })
    }
}

impl Iterator for Preorder {
    type Item = WalkEvent<SyntaxNode>;

    fn next(&mut self) -> Option<WalkEvent<SyntaxNode>> {
        if self.skip_subtree {
            self.do_skip();
            self.skip_subtree = false;
        }
        let next = self.next.take();
        self.next = next.as_ref().and_then(|next| {
            Some(match next {
                WalkEvent::Enter(node) => match node.first_child() {
                    Some(child) => WalkEvent::Enter(child),
                    None => WalkEvent::Leave(node.clone()),
                },
                WalkEvent::Leave(node) => {
                    if node == &self.start {
                        return None;
                    }
                    match node.next_sibling() {
                        Some(sibling) => WalkEvent::Enter(sibling),
                        None => WalkEvent::Leave(node.parent()?),
                    }
                }
            })
        });
        next
    }
}

pub struct PreorderWithTokens {
    start: SyntaxElement,
    next: Option<WalkEvent<SyntaxElement>>,
    skip_subtree: bool,
}

impl PreorderWithTokens {
    fn new(start: SyntaxNode) -> PreorderWithTokens {
        let next = Some(WalkEvent::Enter(start.clone().into()));
        PreorderWithTokens { start: start.into(), next, skip_subtree: false }
    }

    pub fn skip_subtree(&mut self) {
        self.skip_subtree = true;
    }

    #[cold]
    fn do_skip(&mut self) {
        self.next = self.next.take().map(|next| match next {
            WalkEvent::Enter(first_child) => WalkEvent::Leave(first_child.parent().unwrap().into()),
            WalkEvent::Leave(parent) => WalkEvent::Leave(parent),
        })
    }
}

impl Iterator for PreorderWithTokens {
    type Item = WalkEvent<SyntaxElement>;

    fn next(&mut self) -> Option<WalkEvent<SyntaxElement>> {
        if self.skip_subtree {
            self.do_skip();
            self.skip_subtree = false;
        }
        let next = self.next.take();
        self.next = next.as_ref().and_then(|next| {
            Some(match next {
                WalkEvent::Enter(el) => match el {
                    NodeOrToken::Node(node) => match node.first_child_or_token() {
                        Some(child) => WalkEvent::Enter(child),
                        None => WalkEvent::Leave(node.clone().into()),
                    },
                    NodeOrToken::Token(token) => WalkEvent::Leave(token.clone().into()),
                },
                WalkEvent::Leave(el) if el == &self.start => return None,
                WalkEvent::Leave(el) => match el.next_sibling_or_token() {
                    Some(sibling) => WalkEvent::Enter(sibling),
                    None => WalkEvent::Leave(el.parent()?.into()),
                },
            })
        });
        next
    }
}
// endregion
