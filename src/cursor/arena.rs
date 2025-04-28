use std::cell::{Cell, RefCell};
use std::ptr::{self, NonNull};
use std::rc::Rc;

use bumpalo::Bump;
use countme::Count;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use text_size::TextSize;
use triomphe::Arc;

use crate::cursor::{Green, NodeData, NodeKey, SyntaxElement, SyntaxNode, SyntaxToken};
use crate::green::{GreenElement, GreenElementInTree, GreenNodeInTree, GreenTree};
use crate::utility_types::{Delta, PtrEqRc};
use crate::{GreenNode, GreenToken, NodeOrToken, sll};

pub(super) struct RedTree {
    green: Arc<GreenTree>,

    /// Invariant: Nodes are never deleted (only when the whole tree drops).
    arena: Bump,

    /// Invariant: never changes after RedTree is created.
    ///
    /// If this is `true`, the `GreenTree` is unique to this `RedTree` (not
    /// shared with any other tree).
    mutable: bool,

    /// A map from the `GreenNode`/`GreenToken` pointer to the `NodeData`.
    /// Used for reusing nodes, because we never free nodes, so without this
    /// memory usage may grow unexpectedly.
    ///
    /// This is only used for immutable trees; mutable trees use the sll to deduplicate nodes.
    #[expect(unused)]
    allocated_map: RefCell<FxHashMap<NodeKey, NonNull<NodeData>>>,

    /// See [`GreenTree::dependencies`].
    dependencies: RefCell<FxHashSet<PtrEqRc<RedTree>>>,
}

// I haven't done any kind of benchmarking to get to this number. All I know that using a pretty big
// number will make things faster, and memory usage doesn't matter a lot because red trees are usually
// short lived.
const INITIAL_CAPACITY: usize = 200;

impl RedTree {
    #[inline]
    pub(super) fn new(root: GreenNode) -> SyntaxNode {
        let (green_node, green_tree) = root.into_raw_parts();
        let arena = Rc::new(RedTree {
            green: green_tree,
            arena: Bump::with_capacity(INITIAL_CAPACITY * size_of::<NodeData>()),
            mutable: false,
            allocated_map: RefCell::new(FxHashMap::with_capacity_and_hasher(
                INITIAL_CAPACITY,
                FxBuildHasher,
            )),
            dependencies: RefCell::default(),
        });
        let ptr = unsafe {
            arena.alloc_node(None, 0, TextSize::new(0), Green::Node { ptr: Cell::new(green_node) })
        };
        SyntaxNode { ptr, arena }
    }

    #[inline]
    pub(super) fn new_mut(root: GreenNode) -> SyntaxNode {
        let (root_green_node, root_green_tree) = root.into_raw_parts();
        let our_green_tree = GreenTree::new();
        our_green_tree.add_dependency(root_green_tree);
        let arena = Rc::new(RedTree {
            green: our_green_tree.shareable(),
            arena: Bump::with_capacity(INITIAL_CAPACITY * size_of::<NodeData>()),
            mutable: true,
            allocated_map: RefCell::new(FxHashMap::with_capacity_and_hasher(
                INITIAL_CAPACITY,
                FxBuildHasher,
            )),
            dependencies: RefCell::default(),
        });
        let ptr = unsafe {
            arena.alloc_node(
                None,
                0,
                TextSize::new(0),
                Green::Node { ptr: Cell::new(root_green_node) },
            )
        };
        SyntaxNode { ptr, arena }
    }

    #[inline]
    fn ensure_green_tree_liveness(&self, tree: Arc<GreenTree>) {
        if !Arc::ptr_eq(&self.green, &tree) {
            self.green.add_dependency(tree);
        }
    }

    #[inline]
    fn ensure_red_tree_liveness(&self, tree: Rc<RedTree>) {
        if !ptr::eq(self, Rc::as_ptr(&tree)) {
            self.ensure_green_tree_liveness(tree.green.clone());
            self.dependencies.borrow_mut().insert(PtrEqRc(tree));
        }
    }

    /// # Safety
    ///
    /// You need to make sure `green` comes from `self.green` or one of the secondary green trees.
    #[inline]
    pub(super) unsafe fn alloc_node(
        &self,
        parent: Option<NonNull<NodeData>>,
        index: u32,
        offset: TextSize,
        green: Green,
    ) -> NonNull<NodeData> {
        let res = NodeData {
            _c: Count::new(),
            parent: Cell::new(parent),
            index: Cell::new(index),
            green,

            offset,
            first: Cell::new(ptr::null()),
            next: Cell::new(ptr::null()),
            prev: Cell::new(ptr::null()),
        };

        if self.mutable {
            return unsafe {
                let res_ptr: *const NodeData = &res;
                match sll::init(parent.map(|it| &it.as_ref().first), res_ptr.as_ref().unwrap()) {
                    sll::AddToSllResult::AlreadyInSll(node) => {
                        if cfg!(debug_assertions) {
                            assert_eq!((*node).index(), (*res_ptr).index());
                            match ((*node).green(), (*res_ptr).green()) {
                                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => {
                                    assert_eq!(
                                        lhs.as_ptr(),
                                        rhs.as_ptr(),
                                        "lhs={lhs:#?}\n\nrhs={rhs:#?}"
                                    )
                                }
                                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => {
                                    assert_eq!(
                                        lhs.as_ptr(),
                                        rhs.as_ptr(),
                                        "lhs={lhs:#?}\n\nrhs={rhs:#?}"
                                    )
                                }
                                it => {
                                    panic!("node/token confusion: {:?}", it)
                                }
                            }
                        }

                        let res = node as *mut NodeData;
                        NonNull::new_unchecked(res)
                    }
                    it => {
                        let res = NonNull::from(self.arena.alloc(res));
                        it.add_to_sll(res.as_ptr());
                        res
                    }
                }
            };
        }

        // *self
        //     .allocated_map
        //     .borrow_mut()
        //     .entry(res.key(false))
        //     .or_insert_with(|| NonNull::from(self.arena.alloc(res)))
        NonNull::from(self.arena.alloc(res))
    }

    #[inline]
    fn remove_child(&self, node: GreenNodeInTree, index: usize) -> GreenNodeInTree {
        assert!(self.mutable);
        // SAFETY: We have mutable tree access, and that means `self.green` is unique to us
        // (others may read from it, but never write).
        unsafe { node.remove_child(index, &self.green) }
    }

    #[inline]
    fn insert_child(
        &self,
        node: GreenNodeInTree,
        index: usize,
        new_child: GreenElementInTree,
    ) -> GreenNodeInTree {
        assert!(self.mutable);
        // SAFETY: We have mutable tree access, and that means `self.green` is unique to us
        // (others may read from it, but never write).
        unsafe { node.insert_child(index, new_child, &self.green) }
    }

    #[inline]
    fn replace_child(
        &self,
        node: GreenNodeInTree,
        index: usize,
        new_child: GreenElementInTree,
    ) -> GreenNodeInTree {
        assert!(self.mutable);
        // SAFETY: We have mutable tree access, and that means `self.green` is unique to us
        // (others may read from it, but never write).
        unsafe { node.replace_child(index, new_child, &self.green) }
    }

    #[inline]
    pub(super) fn mutable(&self) -> bool {
        self.mutable
    }

    pub(super) fn detach(&self, node: &NodeData) {
        assert!(self.mutable);
        let parent_ptr = match node.parent.take() {
            Some(parent) => parent,
            None => return,
        };

        sll::adjust(node, node.index() + 1, Delta::Sub(1));
        let parent = unsafe { parent_ptr.as_ref() };
        sll::unlink(&parent.first, node);

        match &parent.green {
            Green::Node { ptr } => {
                let green = ptr.get();
                let green = self.remove_child(green, node.index() as usize);
                unsafe { self.respine(parent, green) }
            }
            Green::Token { .. } => unreachable!(),
        }
    }

    pub(super) fn attach_node_child(
        &self,
        parent: NonNull<NodeData>,
        index: usize,
        child: SyntaxElement,
    ) {
        assert!(self.mutable && child.parent().is_none());

        let parent_data = unsafe { parent.as_ref() };

        if !parent_data.first.get().is_null() {
            sll::adjust(unsafe { &*parent_data.first.get() }, index as u32, Delta::Add(1));
        }

        let child_green = {
            let child_data = child.data();
            let child_arena = match &child {
                NodeOrToken::Node(node) => &node.arena,
                NodeOrToken::Token(token) => &token.arena,
            };
            assert!(child_arena.mutable);
            self.ensure_red_tree_liveness(child_arena.clone());

            child_data.parent.set(Some(parent));
            child_data.index.set(index as u32);

            match sll::link(&parent_data.first, child_data) {
                sll::AddToSllResult::AlreadyInSll(_) => {
                    panic!("Child already in sorted linked list")
                }
                it => it.add_to_sll(child.data_ptr().as_ptr()),
            }

            child_data.green()
        };

        match &parent_data.green {
            Green::Node { ptr } => {
                let parent_green = ptr.get();
                let green = self.insert_child(parent_green, index, child_green);
                unsafe { self.respine(parent.as_ref(), green) };
            }
            Green::Token { .. } => unreachable!(),
        }
    }

    unsafe fn respine(&self, mut node: &NodeData, mut new_green: GreenNodeInTree) {
        unsafe {
            loop {
                match &node.green {
                    Green::Node { ptr } => ptr.set(new_green),
                    Green::Token { .. } => unreachable!(),
                };
                match node.parent.get() {
                    Some(parent) => match &parent.as_ref().green {
                        Green::Node { ptr: parent_green } => {
                            let parent_green = parent_green.get();
                            new_green = self.replace_child(
                                parent_green,
                                node.index() as usize,
                                new_green.into(),
                            );
                            node = parent.as_ref();
                        }
                        _ => unreachable!(),
                    },
                    None => break,
                }
            }
        }
    }

    #[inline]
    pub(super) fn green_node(&self, node: &SyntaxNode) -> GreenNode {
        node.data().green.as_node().unwrap().to_green_node(self.green.clone())
    }

    #[inline]
    pub(super) fn green_token(&self, token: &SyntaxToken) -> GreenToken {
        token.data().green.as_token().unwrap().to_green_token(self.green.clone())
    }

    pub(super) fn replace_with(
        &self,
        element: SyntaxElement,
        replacement: GreenElement,
    ) -> GreenNode {
        let (replacement, replacement_tree) = match replacement {
            NodeOrToken::Node(node) => {
                let (node, tree) = node.into_raw_parts();
                (NodeOrToken::Node(node), tree)
            }
            NodeOrToken::Token(token) => {
                let (token, tree) = token.into_raw_parts();
                (NodeOrToken::Token(token), tree)
            }
        };

        if self.mutable {
            self.ensure_green_tree_liveness(replacement_tree);
            // SAFETY: We have mutable access to the tree.
            let green = unsafe { Self::replace_with_impl(&self.green, element, replacement) };
            green.to_green_node(self.green.clone())
        } else {
            let green_tree = GreenTree::new();
            green_tree.add_dependency(replacement_tree);
            // SAFETY: We created the green tree, nobody else can access it.
            let green = unsafe { Self::replace_with_impl(&green_tree, element, replacement) };
            green.to_green_node(green_tree.shareable())
        }
    }

    /// # Safety
    ///
    /// You must ensure no concurrent allocations on `arena`.
    #[inline]
    unsafe fn replace_with_impl(
        arena: &GreenTree,
        element: SyntaxElement,
        replacement: GreenElementInTree,
    ) -> GreenNodeInTree {
        match element.parent() {
            None => *replacement.as_node().unwrap(),
            Some(parent) => {
                // SAFETY: Our precondition.
                let new_parent = unsafe {
                    parent.green_in_tree().replace_child(
                        element.data().index() as usize,
                        replacement,
                        arena,
                    )
                };
                // SAFETY: Our precondition.
                unsafe { Self::replace_with_impl(arena, parent.into(), new_parent.into()) }
            }
        }
    }
}
