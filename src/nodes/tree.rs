use std::sync::Arc;

use crate::{
    maybe_dangling::MaybeDangling,
    nodes::{node::Nodes, token::Token},
};

#[derive(Clone)]
pub(crate) struct Tree(pub(crate) Arc<TreeInner>);

pub(crate) struct TreeInner {
    pub(crate) text: MaybeDangling<Box<str>>,
    /// Always starts with a fake token.
    pub(crate) tokens: MaybeDangling<Box<[Token]>>,
    pub(crate) nodes: Nodes,
}
