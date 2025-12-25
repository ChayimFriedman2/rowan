//! Example that takes the input
//! 1 + 2 * 3 + 4
//! and builds the tree
//! - Marker(Root)
//!   - Marker(Operation)
//!     - Marker(Operation)
//!       - "1" Token(Number)
//!       - "+" Token(Add)
//!       - Marker(Operation)
//!         - "2" Token(Number)
//!         - "*" Token(Mul)
//!         - "3" Token(Number)
//!     - "+" Token(Add)
//!     - "4" Token(Number)

mod ast {
    use crate::{SyntaxKind, SyntaxNode, SyntaxToken};

    #[derive(Clone, Copy)]
    pub enum Operand<'a> {
        Number(SyntaxToken<'a>),
        Operation(Operation<'a>),
    }

    #[derive(Clone, Copy)]
    pub struct Operation<'a>(SyntaxNode<'a>);

    impl<'a> Operation<'a> {
        pub fn cast(node: SyntaxNode<'a>) -> Option<Self> {
            if node.kind() == SyntaxKind::OPERATION { Some(Self(node)) } else { None }
        }

        pub fn op(self) -> SyntaxToken<'a> {
            self.0.token_at(1)
        }

        fn operand(self, idx: usize) -> Operand<'a> {
            if let Some(number) = self.0.try_token_at(idx) {
                Operand::Number(number)
            } else {
                Operand::Operation(Operation::cast(self.0.node_at(idx)).unwrap())
            }
        }

        pub fn lhs(self) -> Operand<'a> {
            self.operand(0)
        }

        pub fn rhs(self) -> Operand<'a> {
            self.operand(2)
        }
    }
}

use rowan::{NodeOrToken, TextSize, TreeBuilder};
use std::iter::Peekable;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(non_camel_case_types)]
#[repr(u16)]
enum SyntaxKind {
    WHITESPACE = 0,

    ADD,
    SUB,
    MUL,
    DIV,

    NUMBER,
    ERROR,
    OPERATION,
    ROOT,
}

use SyntaxKind::*;

impl From<SyntaxKind> for rowan::SyntaxKind {
    fn from(kind: SyntaxKind) -> Self {
        Self(kind as u16)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Lang {}
impl rowan::Language for Lang {
    type Kind = SyntaxKind;
    fn kind_from_raw(raw: rowan::SyntaxKind) -> Self::Kind {
        assert!(raw.0 <= ROOT as u16);
        unsafe { std::mem::transmute::<u16, SyntaxKind>(raw.0) }
    }
    fn kind_to_raw(kind: Self::Kind) -> rowan::SyntaxKind {
        kind.into()
    }
}

type SyntaxNode<'a> = rowan::SyntaxNode<'a, Lang>;
#[allow(unused)]
type SyntaxToken<'a> = rowan::SyntaxToken<'a, Lang>;
#[allow(unused)]
type SyntaxElement<'a> = rowan::SyntaxElement<'a, Lang>;

struct Parser<I: Iterator<Item = (SyntaxKind, String)>> {
    builder: TreeBuilder,
    iter: Peekable<I>,
}
impl<I: Iterator<Item = (SyntaxKind, String)>> Parser<I> {
    fn peek(&mut self) -> Option<SyntaxKind> {
        while self.iter.peek().map(|&(t, _)| t == WHITESPACE).unwrap_or(false) {
            self.bump();
        }
        self.iter.peek().map(|&(t, _)| t)
    }
    fn bump(&mut self) {
        if let Some((token, string)) = self.iter.next() {
            self.builder.token(
                std::iter::empty(),
                token.into(),
                TextSize::of(&string),
                std::iter::empty(),
            );
        }
    }
        match self.peek() {
    fn parse_val(&mut self) {
            Some(NUMBER) => self.bump(),
            _ => {
                self.builder.start_node(ERROR.into());
                self.bump();
                self.builder.finish_node();
            }
        }
    }
    fn handle_operation(&mut self, tokens: &[SyntaxKind], next: fn(&mut Self)) {
        let checkpoint = self.builder.checkpoint();
        next(self);
        while self.peek().map(|t| tokens.contains(&t)).unwrap_or(false) {
            self.builder.start_node_at(checkpoint, OPERATION.into());
            self.bump();
            next(self);
            self.builder.finish_node();
        }
    }
    fn parse_mul(&mut self) {
        self.handle_operation(&[MUL, DIV], Self::parse_val)
    }
    fn parse_add(&mut self) {
        self.handle_operation(&[ADD, SUB], Self::parse_mul)
    }
    fn parse(mut self) -> SyntaxNode {
        self.builder.start_node(ROOT.into());
        self.parse_add();
        self.builder.finish_node();

        SyntaxNode::new_root(self.builder.finish())
    }
}

fn print(indent: usize, element: SyntaxElement) {
    let kind: SyntaxKind = element.kind();
    print!("{:indent$}", "", indent = indent);
    match element {
        NodeOrToken::Node(node) => {
            println!("- {:?}", kind);
            for child in node.children_with_tokens() {
                print(indent + 2, child);
            }
        }

        NodeOrToken::Token(token) => println!("- {:?} {:?}", token.text(), kind),
    }
}

fn main() {
    let ast = Parser {
        builder: GreenNodeBuilder::new(),
        iter: vec![
            // 1 + 2 * 3 + 4
            (NUMBER, "1".into()),
            (WHITESPACE, " ".into()),
            (ADD, "+".into()),
            (WHITESPACE, " ".into()),
            (NUMBER, "2".into()),
            (WHITESPACE, " ".into()),
            (MUL, "*".into()),
            (WHITESPACE, " ".into()),
            (NUMBER, "3".into()),
            (WHITESPACE, " ".into()),
            (ADD, "+".into()),
            (WHITESPACE, " ".into()),
            (NUMBER, "4".into()),
        ]
        .into_iter()
        .peekable(),
    }
    .parse();
    print(0, ast.into());
}
