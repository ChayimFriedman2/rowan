use hashbrown::hash_map::RawEntryMut;
use rustc_hash::FxHasher;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use text_size::TextSize;
use triomphe::UniqueArc;

use crate::green::GreenChild;
use crate::green::arena::GreenTree;
use crate::green::element::GreenElementInTree;
use crate::green::node::GreenNodeInTree;
use crate::green::token::GreenTokenInTree;
use crate::{NodeOrToken, SyntaxKind};

type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Debug)]
struct NoHash<T>(T);

/// Interner for GreenTokens and GreenNodes
// XXX: the impl is a bit tricky. As usual when writing interners, we want to
// store all values in one HashSet.
//
// However, hashing trees is fun: hash of the tree is recursively defined. We
// maintain an invariant -- if the tree is interned, then all of its children
// are interned as well.
//
// That means that computing the hash naively is wasteful -- we just *know*
// hashes of children, and we can re-use those.
//
// So here we use *raw* API of hashbrown and provide the hashes manually,
// instead of going via a `Hash` impl. Our manual `Hash` and the
// `#[derive(Hash)]` are actually different! At some point we had a fun bug,
// where we accidentally mixed the two hashes, which made the cache much less
// efficient.
//
// To fix that, we additionally wrap the data in `NoHash` wrapper, to make sure
// we don't accidentally use the wrong hash!
pub struct NodeCache {
    nodes: HashMap<NoHash<GreenNodeInTree>, ()>,
    tokens: HashMap<NoHash<GreenTokenInTree>, ()>,
    pub(super) arena: UniqueArc<GreenTree>,
}

impl Default for NodeCache {
    #[inline]
    fn default() -> Self {
        Self { nodes: HashMap::default(), tokens: HashMap::default(), arena: GreenTree::new() }
    }
}

fn token_hash(token: &GreenTokenInTree) -> u64 {
    let mut h = FxHasher::default();
    token.kind().hash(&mut h);
    token.text().hash(&mut h);
    h.finish()
}

fn node_hash(node: &GreenNodeInTree) -> u64 {
    let mut h = FxHasher::default();
    node.kind().hash(&mut h);
    for child in node.children() {
        match child {
            GreenChild::Node { rel_offset: _, node } => node_hash(node),
            GreenChild::Token { rel_offset: _, token } => token_hash(token),
        }
        .hash(&mut h)
    }
    h.finish()
}

fn element_id(elem: NodeOrToken<&GreenNodeInTree, &GreenTokenInTree>) -> *const () {
    match elem {
        NodeOrToken::Node(it) => it.data.as_ptr().cast(),
        NodeOrToken::Token(it) => it.data.as_ptr().cast(),
    }
}

impl NodeCache {
    pub(crate) fn node(
        &mut self,
        kind: SyntaxKind,
        children: &mut Vec<(u64, GreenElementInTree)>,
        first_child: usize,
    ) -> (u64, GreenNodeInTree) {
        let mut build_node = |children: &mut Vec<(u64, GreenElementInTree)>| {
            let text_len = children[first_child..].iter().map(|(_, child)| child.text_len()).sum();

            let mut rel_offset = TextSize::new(0);
            let children = children.drain(first_child..).map(|(_, child)| match child {
                NodeOrToken::Node(node) => {
                    let offset = rel_offset;
                    rel_offset += node.text_len();
                    GreenChild::Node { rel_offset: offset, node }
                }
                NodeOrToken::Token(token) => {
                    let offset = rel_offset;
                    rel_offset += token.text_len();
                    GreenChild::Token { rel_offset: offset, token }
                }
            });

            self.arena.alloc_node(kind, text_len, children.len(), children)
        };

        let children_ref = &children[first_child..];
        if children_ref.len() > 3 {
            let node = build_node(children);
            return (0, node);
        }

        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            for &(hash, _) in children_ref {
                if hash == 0 {
                    let node = build_node(children);
                    return (0, node);
                }
                hash.hash(&mut h);
            }
            h.finish()
        };

        // Green nodes are fully immutable, so it's ok to deduplicate them.
        // This is the same optimization that Roslyn does
        // https://github.com/KirillOsenkov/Bliki/wiki/Roslyn-Immutable-Trees
        //
        // For example, all `#[inline]` in this file share the same green node!
        // For `libsyntax/parse/parser.rs`, measurements show that deduping saves
        // 17% of the memory for green nodes!
        let entry = self.nodes.raw_entry_mut().from_hash(hash, |node| {
            node.0.kind() == kind && node.0.children().len() == children_ref.len() && {
                let lhs = node.0.children();
                let rhs = children_ref.iter().map(|(_, it)| it);

                let lhs = lhs
                    .iter()
                    .map(|it| match it {
                        GreenChild::Node { rel_offset: _, node } => NodeOrToken::Node(node),
                        GreenChild::Token { rel_offset: _, token } => NodeOrToken::Token(token),
                    })
                    .map(element_id);
                let rhs = rhs.map(|it| element_id(it.as_ref()));

                lhs.eq(rhs)
            }
        });

        let node = match entry {
            RawEntryMut::Occupied(entry) => {
                drop(children.drain(first_child..));
                entry.key().0
            }
            RawEntryMut::Vacant(entry) => {
                let node = build_node(children);
                entry.insert_with_hasher(hash, NoHash(node), (), |n| node_hash(&n.0));
                node
            }
        };

        (hash, node)
    }

    pub(crate) fn token(&mut self, kind: SyntaxKind, text: &str) -> (u64, GreenTokenInTree) {
        let hash = {
            let mut h = FxHasher::default();
            kind.hash(&mut h);
            text.hash(&mut h);
            h.finish()
        };

        let entry = self
            .tokens
            .raw_entry_mut()
            .from_hash(hash, |token| token.0.kind() == kind && token.0.text() == text);

        let token = match entry {
            RawEntryMut::Occupied(entry) => entry.key().0,
            RawEntryMut::Vacant(entry) => {
                let token = self.arena.alloc_token(kind, text);
                entry.insert_with_hasher(hash, NoHash(token), (), |t| token_hash(&t.0));
                token
            }
        };

        (hash, token)
    }
}
