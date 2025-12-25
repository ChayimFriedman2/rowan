mod tree;
mod node;
mod token;

pub(crate) use node::{ChildKind, List, Node, NodeOrListOrToken, Nodes};
pub(crate) use token::{AttachedTrivia, Token, TokenRef, TokenRefIter};
pub(crate) use tree::{Tree, TreeInner};

/// SyntaxKind is a type tag for each token or node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxKind(pub u16);
