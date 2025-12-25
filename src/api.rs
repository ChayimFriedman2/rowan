use std::{fmt, marker::PhantomData};

use text_size::{TextRange, TextSize};

use crate::{
    nodes::{ChildKind, List, Node, SyntaxKind, TokenRef, TokenRefIter, Tree, TreeInner},
    utility_types::{NodeOrToken, SyntaxElement, TokenAtOffset},
};

pub trait Language: Sized + Copy + fmt::Debug + Eq + Ord + std::hash::Hash {
    type Kind: Sized + Copy + fmt::Debug + Eq + Ord + std::hash::Hash;

    fn kind_from_raw(raw: SyntaxKind) -> Self::Kind;
    fn kind_to_raw(kind: Self::Kind) -> SyntaxKind;
}

pub struct SyntaxTree<L> {
    pub(crate) tree: Tree,
    pub(crate) _marker: PhantomData<L>,
}

impl<L> Clone for SyntaxTree<L> {
    #[inline]
    fn clone(&self) -> Self {
        Self { tree: self.tree.clone(), _marker: PhantomData }
    }
}

impl<L: Language> SyntaxTree<L> {
    #[inline]
    pub fn root(&self) -> SyntaxNode<'_, L> {
        SyntaxNode { tree: &self.tree.0, node: self.tree.0.nodes.root(), _marker: PhantomData }
    }

    #[inline]
    pub fn text(&self) -> &str {
        &self.tree.0.text
    }
}

pub struct SyntaxToken<'a, L> {
    tree: &'a TreeInner,
    token: TokenRef<'a>,
    _marker: PhantomData<L>,
}

impl<L> Clone for SyntaxToken<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for SyntaxToken<'_, L> {}

impl<'a, L: Language> SyntaxToken<'a, L> {
    #[inline]
    pub fn kind(self) -> L::Kind {
        L::kind_from_raw(self.token.get().kind)
    }

    #[inline]
    pub fn text_range(self) -> TextRange {
        self.token.text_range()
    }

    #[inline]
    pub fn prev_token(self) -> Option<SyntaxToken<'a, L>> {
        Some(SyntaxToken {
            tree: self.tree,
            token: self.token.prev_token(self.tree)?,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn next_token(self) -> Option<SyntaxToken<'a, L>> {
        Some(SyntaxToken {
            tree: self.tree,
            token: self.token.next_token(self.tree)?,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn leading_trivia(self) -> TriviaIter<'a, L> {
        TriviaIter { tree: self.tree, tokens: self.token.leading_trivia(), _marker: PhantomData }
    }

    #[inline]
    pub fn trailing_trivia(self) -> TriviaIter<'a, L> {
        TriviaIter { tree: self.tree, tokens: self.token.trailing_trivia(), _marker: PhantomData }
    }

    #[inline]
    pub fn text(self) -> &'a str {
        self.token.text(self.tree)
    }

    #[inline]
    pub fn text_range_including_trivia(self) -> TextRange {
        let first_token = self.leading_trivia().next().unwrap_or(self);
        let last_token = self.trailing_trivia().next_back().unwrap_or(self);
        TextRange::new(first_token.token.start(), last_token.token.end())
    }

    #[inline]
    pub fn text_including_trivia(self) -> &'a str {
        &self.tree.text[self.text_range_including_trivia()]
    }

    #[inline]
    pub fn parent(self) -> SyntaxNode<'a, L> {
        SyntaxNode { tree: self.tree, node: self.token.parent(), _marker: PhantomData }
    }

    #[inline]
    pub fn parent_ancestors(self) -> impl Iterator<Item = SyntaxNode<'a, L>> + Clone {
        self.parent().ancestors()
    }
}

pub struct TriviaIter<'a, L> {
    tree: &'a TreeInner,
    tokens: TokenRefIter<'a>,
    _marker: PhantomData<L>,
}

impl<L> Clone for TriviaIter<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        Self { tree: self.tree, tokens: self.tokens.clone(), _marker: PhantomData }
    }
}

impl<'a, L: Language> Iterator for TriviaIter<'a, L> {
    type Item = SyntaxToken<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(SyntaxToken { tree: self.tree, token: self.tokens.next()?, _marker: PhantomData })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.tokens.len();
        (len, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }
}

impl<'a, L: Language> DoubleEndedIterator for TriviaIter<'a, L> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(SyntaxToken { tree: self.tree, token: self.tokens.next_back()?, _marker: PhantomData })
    }
}

impl<L: Language> ExactSizeIterator for TriviaIter<'_, L> {
    #[inline]
    fn len(&self) -> usize {
        self.tokens.len()
    }
}

pub struct SyntaxNode<'a, L> {
    tree: &'a TreeInner,
    node: &'a Node,
    _marker: PhantomData<L>,
}

impl<L> Clone for SyntaxNode<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for SyntaxNode<'_, L> {}

impl<'a, L: Language> SyntaxNode<'a, L> {
    #[inline]
    pub fn kind(self) -> L::Kind {
        L::kind_from_raw(self.node.kind)
    }

    #[inline]
    pub fn first_token(self) -> SyntaxToken<'a, L> {
        SyntaxToken {
            tree: self.tree,
            token: self.node.first_token(self.tree),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn last_token(self) -> SyntaxToken<'a, L> {
        SyntaxToken {
            tree: self.tree,
            token: self.node.last_token(self.tree),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn text_range(self) -> TextRange {
        self.node.text_range(self.tree)
    }

    #[inline]
    pub fn text(self) -> &'a str {
        self.node.text(self.tree)
    }

    #[inline]
    pub fn parent(self) -> Option<SyntaxNode<'a, L>> {
        Some(SyntaxNode { tree: self.tree, node: self.node.parent()?, _marker: PhantomData })
    }

    #[inline]
    pub fn try_token_at(self, idx: usize) -> Option<SyntaxToken<'a, L>> {
        Some(SyntaxToken {
            tree: self.tree,
            token: self.node.children().get(idx)?.as_token()?,
            _marker: PhantomData,
        })
    }

    #[inline]
    #[track_caller]
    pub fn token_at(self, idx: usize) -> SyntaxToken<'a, L> {
        match self.try_token_at(idx) {
            Some(it) => it,
            None => expected_token(idx),
        }
    }

    #[inline]
    pub fn try_node_at(self, idx: usize) -> Option<SyntaxNode<'a, L>> {
        Some(SyntaxNode {
            tree: self.tree,
            node: self.node.children().get(idx)?.as_node()?,
            _marker: PhantomData,
        })
    }

    #[inline]
    #[track_caller]
    pub fn node_at(self, idx: usize) -> SyntaxNode<'a, L> {
        match self.try_node_at(idx) {
            Some(it) => it,
            None => expected_node(idx),
        }
    }

    #[inline]
    pub fn try_list_at(self, idx: usize) -> Option<SyntaxList<'a, L>> {
        Some(SyntaxList {
            tree: self.tree,
            list: self.node.children().get(idx)?.as_list()?,
            _marker: PhantomData,
        })
    }

    #[inline]
    #[track_caller]
    pub fn list_at(self, idx: usize) -> SyntaxList<'a, L> {
        match self.try_list_at(idx) {
            Some(it) => it,
            None => expected_list(idx),
        }
    }

    #[inline]
    pub fn try_child_at(self, idx: usize) -> Option<NodeOrTokenOrList<'a, L>> {
        self.node.children().get(idx).map(|child| self.map_child_kind(child.kind()))
    }

    #[inline]
    #[track_caller]
    pub fn child_at(self, idx: usize) -> NodeOrTokenOrList<'a, L> {
        self.map_child_kind(self.node.children()[idx].kind())
    }

    #[inline]
    fn map_child_kind(self, child: ChildKind<'a>) -> NodeOrTokenOrList<'a, L> {
        match child {
            ChildKind::Token(token) => NodeOrTokenOrList::Token(SyntaxToken {
                tree: self.tree,
                token,
                _marker: PhantomData,
            }),
            ChildKind::Node(node) => {
                NodeOrTokenOrList::Node(SyntaxNode { tree: self.tree, node, _marker: PhantomData })
            }
            ChildKind::List(list) => {
                NodeOrTokenOrList::List(SyntaxList { list, tree: self.tree, _marker: PhantomData })
            }
        }
    }

    #[inline]
    pub fn ancestors(self) -> impl Iterator<Item = SyntaxNode<'a, L>> + Clone {
        std::iter::successors(Some(self), |it| it.parent())
    }

    #[inline]
    pub fn children_with_tokens_and_lists(self) -> ChildrenWithTokensAndLists<'a, L> {
        ChildrenWithTokensAndLists {
            tree: self.tree,
            children: self.node.children().iter(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn children_with_lists(self) -> ChildrenWithLists<'a, L> {
        ChildrenWithLists { inner: self.children_with_tokens_and_lists() }
    }

    #[inline]
    pub fn children_with_tokens(self) -> ChildrenWithTokens<'a, L> {
        ChildrenWithTokens {
            active_list_iter: None,
            children: self.children_with_tokens_and_lists(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn children(self) -> Children<'a, L> {
        Children { inner: self.children_with_tokens() }
    }

    #[inline]
    pub fn preorder(self) -> Preorder<'a, L> {
        Preorder::new(self)
    }

    #[inline]
    pub fn preorder_with_tokens(self) -> PreorderWithTokens<'a, L> {
        PreorderWithTokens::new(self)
    }

    #[inline]
    pub fn descendants(&self) -> impl Iterator<Item = SyntaxNode<'a, L>> + Clone {
        self.preorder_with_tokens().filter_map(|event| match event {
            WalkEventWithTokens::EnterNode(node) => Some(node),
            WalkEventWithTokens::LeaveNode(_) | WalkEventWithTokens::Token(_) => None,
        })
    }

    #[inline]
    pub fn descendants_with_tokens(&self) -> impl Iterator<Item = SyntaxElement<'a, L>> + Clone {
        self.preorder_with_tokens().filter_map(|event| match event {
            WalkEventWithTokens::EnterNode(node) => Some(SyntaxElement::Node(node)),
            WalkEventWithTokens::Token(node) => Some(SyntaxElement::Token(node)),
            WalkEventWithTokens::LeaveNode(_) => None,
        })
    }

    #[inline]
    pub fn token_at_offset(self, offset: TextSize) -> TokenAtOffset<SyntaxToken<'a, L>> {
        self.node.token_at_offset(self.tree, offset).map(|token| SyntaxToken {
            tree: self.tree,
            token,
            _marker: PhantomData,
        })
    }

    #[inline]
    pub fn covering_element(self, range: TextRange) -> SyntaxElement<'a, L> {
        match self.node.covering_element(self.tree, range) {
            NodeOrToken::Node(node) => {
                NodeOrToken::Node(SyntaxNode { tree: self.tree, node, _marker: PhantomData })
            }
            NodeOrToken::Token(token) => {
                NodeOrToken::Token(SyntaxToken { tree: self.tree, token, _marker: PhantomData })
            }
        }
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn expected_token(idx: usize) -> ! {
    panic!("expected a token at index {idx}")
}

#[cold]
#[inline(never)]
#[track_caller]
fn expected_node(idx: usize) -> ! {
    panic!("expected a node at index {idx}")
}

#[cold]
#[inline(never)]
#[track_caller]
fn expected_list(idx: usize) -> ! {
    panic!("expected a list at index {idx}")
}

pub struct SyntaxList<'a, L> {
    list: &'a List,
    tree: &'a TreeInner,
    _marker: PhantomData<L>,
}

impl<L> Clone for SyntaxList<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for SyntaxList<'_, L> {}

impl<'a, L: Language> SyntaxList<'a, L> {
    #[inline]
    pub fn len(self) -> usize {
        self.list.children().len()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(self, idx: usize) -> Option<SyntaxNode<'a, L>> {
        Some(SyntaxNode {
            tree: self.tree,
            node: self.list.children().get(idx)?,
            _marker: PhantomData,
        })
    }

    // Implementing `Index` isn't possible due to trait limitations.
    #[inline]
    #[track_caller]
    pub fn at(self, idx: usize) -> SyntaxNode<'a, L> {
        SyntaxNode { tree: self.tree, node: &self.list.children()[idx], _marker: PhantomData }
    }

    #[inline]
    pub fn iter(self) -> SyntaxListIter<'a, L> {
        SyntaxListIter { tree: self.tree, iter: self.list.children().iter(), _marker: PhantomData }
    }
}

pub struct SyntaxListIter<'a, L> {
    tree: &'a TreeInner,
    iter: std::slice::Iter<'a, &'a Node>,
    _marker: PhantomData<L>,
}

impl<L> Clone for SyntaxListIter<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        Self { tree: self.tree, iter: self.iter.clone(), _marker: PhantomData }
    }
}

impl<'a, L: Language> Iterator for SyntaxListIter<'a, L> {
    type Item = SyntaxNode<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|&node| SyntaxNode { tree: self.tree, node, _marker: PhantomData })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }
}

impl<'a, L: Language> DoubleEndedIterator for SyntaxListIter<'a, L> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|&node| SyntaxNode {
            tree: self.tree,
            node,
            _marker: PhantomData,
        })
    }
}

impl<L: Language> ExactSizeIterator for SyntaxListIter<'_, L> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[derive(Clone, Copy)]
pub enum NodeOrTokenOrList<'a, L> {
    Node(SyntaxNode<'a, L>),
    Token(SyntaxToken<'a, L>),
    List(SyntaxList<'a, L>),
}

pub struct ChildrenWithTokensAndLists<'a, L> {
    tree: &'a TreeInner,
    children: std::slice::Iter<'a, crate::nodes::NodeOrListOrToken>,
    _marker: PhantomData<L>,
}

impl<L> Clone for ChildrenWithTokensAndLists<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        Self { tree: self.tree, children: self.children.clone(), _marker: PhantomData }
    }
}

impl<'a, L: Language> ChildrenWithTokensAndLists<'a, L> {
    #[inline]
    fn map_child(
        &self,
        child: Option<&'a crate::nodes::NodeOrListOrToken>,
    ) -> Option<NodeOrTokenOrList<'a, L>> {
        let tree = self.tree;
        child.map(|child| match child.kind() {
            ChildKind::Token(token) => {
                NodeOrTokenOrList::Token(SyntaxToken { tree, token, _marker: PhantomData })
            }
            ChildKind::Node(node) => {
                NodeOrTokenOrList::Node(SyntaxNode { tree, node, _marker: PhantomData })
            }
            ChildKind::List(list) => {
                NodeOrTokenOrList::List(SyntaxList { list, tree, _marker: PhantomData })
            }
        })
    }
}

impl<'a, L: Language> Iterator for ChildrenWithTokensAndLists<'a, L> {
    type Item = NodeOrTokenOrList<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let child = self.children.next();
        self.map_child(child)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.children.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }
}

impl<'a, L: Language> DoubleEndedIterator for ChildrenWithTokensAndLists<'a, L> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let child = self.children.next_back();
        self.map_child(child)
    }
}

impl<L: Language> ExactSizeIterator for ChildrenWithTokensAndLists<'_, L> {
    #[inline]
    fn len(&self) -> usize {
        self.children.len()
    }
}

pub enum NodeOrList<'a, L> {
    Node(SyntaxNode<'a, L>),
    List(SyntaxList<'a, L>),
}

impl<L> Clone for NodeOrList<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for NodeOrList<'_, L> {}

pub struct ChildrenWithLists<'a, L> {
    inner: ChildrenWithTokensAndLists<'a, L>,
}

impl<L> Clone for ChildrenWithLists<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<'a, L: Language> ChildrenWithLists<'a, L> {
    #[inline]
    fn filter_child(child: NodeOrTokenOrList<'a, L>) -> Option<NodeOrList<'a, L>> {
        match child {
            NodeOrTokenOrList::Node(it) => Some(NodeOrList::Node(it)),
            NodeOrTokenOrList::List(it) => Some(NodeOrList::List(it)),
            NodeOrTokenOrList::Token(_) => None,
        }
    }

    #[inline]
    fn iter(self) -> impl DoubleEndedIterator<Item = NodeOrList<'a, L>> {
        self.inner.filter_map(|child| match child {
            NodeOrTokenOrList::Node(it) => Some(NodeOrList::Node(it)),
            NodeOrTokenOrList::List(it) => Some(NodeOrList::List(it)),
            NodeOrTokenOrList::Token(_) => None,
        })
    }
}

impl<'a, L: Language> Iterator for ChildrenWithLists<'a, L> {
    type Item = NodeOrList<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|child| Self::filter_child(child))
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter().fold(init, f)
    }

    #[inline]
    fn for_each<F>(self, f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        self.iter().for_each(f);
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.next_back()
    }
}

impl<'a, L: Language> DoubleEndedIterator for ChildrenWithLists<'a, L> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.by_ref().rev().find_map(|child| Self::filter_child(child))
    }

    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter().rfold(init, f)
    }
}

pub struct ChildrenWithTokens<'a, L> {
    active_list_iter: Option<SyntaxListIter<'a, L>>,
    children: ChildrenWithTokensAndLists<'a, L>,
    _marker: PhantomData<L>,
}

impl<L> Clone for ChildrenWithTokens<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            active_list_iter: self.active_list_iter.clone(),
            children: self.children.clone(),
            _marker: PhantomData,
        }
    }
}

impl<'a, L: Language> Iterator for ChildrenWithTokens<'a, L> {
    type Item = SyntaxElement<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(list_iter) = &mut self.active_list_iter {
                match list_iter.next() {
                    Some(list_item) => return Some(SyntaxElement::Node(list_item)),
                    None => self.active_list_iter = None,
                }
            }

            match self.children.next()? {
                NodeOrTokenOrList::Node(it) => return Some(SyntaxElement::Node(it)),
                NodeOrTokenOrList::Token(it) => return Some(SyntaxElement::Token(it)),
                NodeOrTokenOrList::List(it) => self.active_list_iter = Some(it.iter()),
            }
        }
    }

    #[inline]
    fn for_each<F>(self, mut f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        self.fold((), |(), item| f(item))
    }

    #[inline]
    fn fold<B, F>(self, mut init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        for child in self.children {
            match child {
                NodeOrTokenOrList::Node(it) => init = f(init, SyntaxElement::Node(it)),
                NodeOrTokenOrList::Token(it) => init = f(init, SyntaxElement::Token(it)),
                NodeOrTokenOrList::List(it) => {
                    for child in it.iter() {
                        init = f(init, SyntaxElement::Node(child))
                    }
                }
            }
        }
        init
    }
}

pub struct Children<'a, L> {
    inner: ChildrenWithTokens<'a, L>,
}

impl<L> Clone for Children<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<'a, L: Language> Children<'a, L> {
    #[inline]
    fn filter_child(child: SyntaxElement<'a, L>) -> Option<SyntaxNode<'a, L>> {
        match child {
            SyntaxElement::Node(it) => Some(it),
            SyntaxElement::Token(_) => None,
        }
    }

    #[inline]
    fn iter(self) -> impl Iterator<Item = SyntaxNode<'a, L>> {
        self.inner.filter_map(|child| match child {
            SyntaxElement::Node(it) => Some(it),
            SyntaxElement::Token(_) => None,
        })
    }
}

impl<'a, L: Language> Iterator for Children<'a, L> {
    type Item = SyntaxNode<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|child| Self::filter_child(child))
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter().fold(init, f)
    }

    #[inline]
    fn for_each<F>(self, f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        self.iter().for_each(f);
    }
}

#[derive(Clone)]
pub struct Preorder<'a, L> {
    inner: PreorderWithTokens<'a, L>,
}

impl<'a, L: Language> Preorder<'a, L> {
    #[inline]
    fn new(start: SyntaxNode<'a, L>) -> Preorder<'a, L> {
        Preorder { inner: PreorderWithTokens::new(start) }
    }

    #[inline]
    pub fn skip_subtree(&mut self) {
        self.inner.skip_subtree();
    }
}

impl<'a, L: Language> Iterator for Preorder<'a, L> {
    type Item = WalkEvent<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.find_map(|item| match item {
            WalkEventWithTokens::EnterNode(it) => Some(WalkEvent::Enter(it)),
            WalkEventWithTokens::LeaveNode(it) => Some(WalkEvent::Leave(it)),
            WalkEventWithTokens::Token(_) => None,
        })
    }
}

pub enum WalkEvent<'a, L> {
    Enter(SyntaxNode<'a, L>),
    Leave(SyntaxNode<'a, L>),
}

impl<L> Clone for WalkEvent<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for WalkEvent<'_, L> {}

#[derive(Clone)]
pub struct PreorderWithTokens<'a, L> {
    stack: Vec<(SyntaxNode<'a, L>, ChildrenWithTokens<'a, L>)>,
    root: Option<SyntaxNode<'a, L>>,
}

impl<'a, L: Language> PreorderWithTokens<'a, L> {
    #[inline]
    fn new(start: SyntaxNode<'a, L>) -> PreorderWithTokens<'a, L> {
        PreorderWithTokens { stack: Vec::with_capacity(128), root: Some(start) }
    }

    #[inline]
    pub fn skip_subtree(&mut self) {
        assert!(self.stack.pop().is_some(), "must have a subtree to skip");
    }
}

impl<'a, L: Language> Iterator for PreorderWithTokens<'a, L> {
    type Item = WalkEventWithTokens<'a, L>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let Some((_, active_node)) = self.stack.last_mut() else {
            let root = self.root?;
            self.root = None;
            return Some(WalkEventWithTokens::EnterNode(root));
        };
        match active_node.next() {
            Some(SyntaxElement::Node(child)) => {
                self.stack.push((child, child.children_with_tokens()));
                Some(WalkEventWithTokens::EnterNode(child))
            }
            Some(SyntaxElement::Token(child)) => Some(WalkEventWithTokens::Token(child)),
            None => {
                let (exited_node, _) = self.stack.pop().expect("should have an exited-from node");
                Some(WalkEventWithTokens::LeaveNode(exited_node))
            }
        }
    }
}

pub enum WalkEventWithTokens<'a, L> {
    EnterNode(SyntaxNode<'a, L>),
    LeaveNode(SyntaxNode<'a, L>),
    Token(SyntaxToken<'a, L>),
}

impl<L> Clone for WalkEventWithTokens<'_, L> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> Copy for WalkEventWithTokens<'_, L> {}

impl<L: Language> fmt::Debug for SyntaxNode<'_, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            let mut level = 0;
            for event in self.preorder_with_tokens() {
                match event {
                    WalkEventWithTokens::EnterNode(node) => {
                        for _ in 0..level {
                            write!(f, "  ")?;
                        }
                        writeln!(f, "{:?}", node)?;
                        level += 1;
                    }
                    WalkEventWithTokens::Token(token) => {
                        for _ in 0..level {
                            write!(f, "  ")?;
                        }
                        writeln!(f, "{:?}", token)?;
                        level += 1;
                    }
                    WalkEventWithTokens::LeaveNode(_) => level -= 1,
                }
            }
            assert_eq!(level, 0);
            Ok(())
        } else {
            write!(f, "{:?}@{:?}", self.kind(), self.text_range())
        }
    }
}

impl<L: Language> fmt::Display for SyntaxNode<'_, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.text(), f)
    }
}

impl<L: Language> fmt::Debug for SyntaxToken<'_, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}@{:?}", self.kind(), self.text_range())?;
        if self.text().len() < 25 {
            return write!(f, " {:?}", self.text());
        }
        let text = self.text();
        for idx in 21..25 {
            if text.is_char_boundary(idx) {
                let text = format!("{} ...", &text[..idx]);
                return write!(f, " {:?}", text);
            }
        }
        unreachable!()
    }
}

impl<L: Language> fmt::Display for SyntaxToken<'_, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.text(), f)
    }
}

impl<L: Language> fmt::Debug for SyntaxList<'_, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, L: Language> From<SyntaxNode<'a, L>> for SyntaxElement<'a, L> {
    #[inline]
    fn from(node: SyntaxNode<'a, L>) -> SyntaxElement<'a, L> {
        NodeOrToken::Node(node)
    }
}

impl<'a, L: Language> From<SyntaxToken<'a, L>> for SyntaxElement<'a, L> {
    #[inline]
    fn from(token: SyntaxToken<'a, L>) -> SyntaxElement<'a, L> {
        NodeOrToken::Token(token)
    }
}
