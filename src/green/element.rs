use crate::{
    NodeOrToken, TextSize,
    green::{GreenNode, GreenToken, SyntaxKind, node::GreenNodeInTree, token::GreenTokenInTree},
};

pub(crate) type GreenElementInTree = NodeOrToken<GreenNodeInTree, GreenTokenInTree>;

impl From<GreenNodeInTree> for GreenElementInTree {
    #[inline]
    fn from(node: GreenNodeInTree) -> Self {
        NodeOrToken::Node(node)
    }
}

impl From<GreenTokenInTree> for GreenElementInTree {
    #[inline]
    fn from(token: GreenTokenInTree) -> Self {
        NodeOrToken::Token(token)
    }
}

impl GreenElementInTree {
    /// Returns kind of this element.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }

    /// Returns the length of the text covered by this element.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        match self {
            NodeOrToken::Node(it) => it.text_len(),
            NodeOrToken::Token(it) => it.text_len(),
        }
    }
}

pub type GreenElement = NodeOrToken<GreenNode, GreenToken>;

impl From<GreenNode> for GreenElement {
    #[inline]
    fn from(node: GreenNode) -> GreenElement {
        NodeOrToken::Node(node)
    }
}

impl From<GreenToken> for GreenElement {
    #[inline]
    fn from(token: GreenToken) -> GreenElement {
        NodeOrToken::Token(token)
    }
}

impl GreenElement {
    /// Returns kind of this element.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        match self {
            NodeOrToken::Node(it) => it.kind(),
            NodeOrToken::Token(it) => it.kind(),
        }
    }

    /// Returns the length of the text covered by this element.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        match self {
            NodeOrToken::Node(it) => it.text_len(),
            NodeOrToken::Token(it) => it.text_len(),
        }
    }
}
