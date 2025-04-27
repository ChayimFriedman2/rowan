mod node;
mod token;
mod element;
mod builder;
mod node_cache;
mod arena;

pub(crate) use self::{
    arena::GreenTree,
    element::GreenElementInTree,
    node::{GreenChild, GreenNodeInTree},
    token::GreenTokenInTree,
};

pub use self::{
    builder::{Checkpoint, GreenNodeBuilder},
    element::GreenElement,
    node::{Children, GreenNode},
    node_cache::NodeCache,
    token::GreenToken,
};

/// SyntaxKind is a type tag for each token or node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxKind(pub u16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_send_sync() {
        fn f<T: Send + Sync>() {}
        f::<GreenNode>();
        f::<GreenToken>();
        f::<GreenElement>();
    }

    #[test]
    fn test_size_of() {
        use std::mem::size_of;

        eprintln!("GreenNode          {}", size_of::<GreenNode>());
        eprintln!("GreenToken         {}", size_of::<GreenToken>());
        eprintln!("GreenElement       {}", size_of::<GreenElement>());
    }
}
