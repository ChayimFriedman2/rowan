//! A generic library for lossless syntax trees.
//! See `examples/s_expressions.rs` for a tutorial.
#![forbid(
    // missing_debug_implementations,
    unconditional_recursion,
    future_incompatible,
    // missing_docs,
)]
#![deny(unsafe_code, rust_2018_idioms)]

mod utility_types;

mod builder;
#[allow(unsafe_code)]
mod nodes;
mod api;

#[allow(unsafe_code)]
mod maybe_dangling;

pub use text_size::{TextLen, TextRange, TextSize};

pub use crate::{
    api::{
        Children, ChildrenWithLists, ChildrenWithTokens, ChildrenWithTokensAndLists, Language,
        NodeOrList, NodeOrTokenOrList, Preorder, PreorderWithTokens, SyntaxList, SyntaxListIter,
        SyntaxNode, SyntaxToken, SyntaxTree, TriviaIter, WalkEvent, WalkEventWithTokens,
    },
    builder::TreeBuilder,
    nodes::SyntaxKind,
    utility_types::{NodeOrToken, SyntaxElement, TokenAtOffset},
};
