use std::{
    alloc::Layout,
    fmt,
    hash::{Hash, Hasher},
    ptr::NonNull,
    slice,
};

use countme::Count;
use triomphe::Arc;

use crate::{
    TextSize,
    green::{SyntaxKind, arena::GreenTree},
};

#[repr(packed)] // Make this 6 bytes instead of 8.
pub(super) struct GreenTokenHead {
    kind: SyntaxKind,
    text_len: TextSize,
    _c: Count<GreenToken>,
}

// The following impls don't include `text_len`, it will be handled as the slice len.
impl Hash for GreenTokenHead {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Self { kind, text_len: _, _c: _ } = *self;
        kind.hash(state);
    }
}

impl PartialEq for GreenTokenHead {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let Self { kind, text_len: _, _c: _ } = *self;
        let Self { kind: other_kind, text_len: _, _c: _ } = *other;
        kind == other_kind
    }
}

impl Eq for GreenTokenHead {}

impl GreenTokenHead {
    #[inline]
    pub(super) fn layout(text_len: usize) -> Layout {
        Layout::new::<GreenTokenHead>()
            .extend(Layout::array::<u8>(text_len).expect("too big node"))
            .expect("too big node")
            .0
            .pad_to_align()
    }

    #[inline]
    pub(super) fn new(kind: SyntaxKind, text: &str) -> Self {
        Self { kind, text_len: TextSize::of(text), _c: Count::new() }
    }
}

pub(super) struct GreenTokenData {
    head: GreenTokenHead,
    text: [u8; 0],
}

#[derive(Clone, Copy)]
pub(crate) struct GreenTokenInTree {
    /// INVARIANT: This points at a valid `GreenTokenInTree` then `str` with len `text_len`,
    /// with `#[repr(C)]`.
    pub(super) data: NonNull<GreenTokenData>,
}

// SAFETY: The pointer is valid.
unsafe impl Send for GreenTokenInTree {}
unsafe impl Sync for GreenTokenInTree {}

impl GreenTokenInTree {
    #[inline]
    fn header(&self) -> &GreenTokenHead {
        // SAFETY: `data`'s invariant.
        unsafe { &*self.header_ptr_mut() }
    }

    /// Does not require the pointer to be valid.
    #[inline]
    pub(super) fn header_ptr_mut(&self) -> *mut GreenTokenHead {
        // SAFETY: `&raw mut` doesn't require the data to be valid, only allocated.
        unsafe { &raw mut (*self.data.as_ptr()).head }
    }

    #[inline]
    pub(crate) fn text(&self) -> &str {
        // SAFETY: `data`'s invariant.
        unsafe {
            std::str::from_utf8_unchecked(slice::from_raw_parts(
                self.text_ptr_mut(),
                self.header().text_len.into(),
            ))
        }
    }

    #[inline]
    pub(super) fn text_ptr_mut(&self) -> *mut u8 {
        // SAFETY: `&raw mut` doesn't require the data to be valid, only allocated.
        unsafe { (&raw mut (*self.data.as_ptr()).text).cast::<u8>() }
    }

    #[inline]
    pub(crate) fn kind(&self) -> SyntaxKind {
        self.header().kind
    }

    #[inline]
    pub(crate) fn text_len(&self) -> TextSize {
        TextSize::of(self.text())
    }

    #[inline]
    pub(crate) fn as_ptr(self) -> NonNull<()> {
        self.data.cast()
    }

    #[inline]
    pub(crate) fn to_green_token(self, arena: Arc<GreenTree>) -> GreenToken {
        GreenToken { token: self, _arena: arena }
    }
}

impl fmt::Debug for GreenTokenInTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreenToken")
            .field("kind", &{ self.header().kind })
            .field("text", &self.text())
            .finish()
    }
}

impl fmt::Display for GreenTokenInTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

impl PartialEq for GreenTokenInTree {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.header() == other.header() && self.text() == other.text()
    }
}

impl Eq for GreenTokenInTree {}

impl Hash for GreenTokenInTree {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.header().hash(state);
        self.text().hash(state);
    }
}

/// Leaf node in the immutable tree.
#[derive(Clone)]
pub struct GreenToken {
    pub(super) token: GreenTokenInTree,
    pub(super) _arena: Arc<GreenTree>,
}

impl PartialEq for GreenToken {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.token == other.token
    }
}

impl Eq for GreenToken {}

impl Hash for GreenToken {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.token.hash(state);
    }
}

impl fmt::Debug for GreenToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.token, f)
    }
}

impl fmt::Display for GreenToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.token, f)
    }
}

impl GreenToken {
    /// Creates a freestanding token.
    ///
    /// Note: this is expensive. Prefer building your token directly into the tree with [`GreenNodeBuilder`].
    ///
    /// [`GreenNodeBuilder`]: crate::GreenNodeBuilder
    #[inline]
    pub fn new(kind: SyntaxKind, text: &str) -> GreenToken {
        let mut arena = GreenTree::new();
        let token = arena.alloc_token(kind, text);
        token.to_green_token(arena.shareable())
    }

    #[inline]
    pub(crate) fn into_raw_parts(self) -> (GreenTokenInTree, Arc<GreenTree>) {
        (self.token, self._arena)
    }

    /// Kind of this Token.
    #[inline]
    pub fn kind(&self) -> SyntaxKind {
        self.token.kind()
    }

    /// Text of this Token.
    #[inline]
    pub fn text(&self) -> &str {
        self.token.text()
    }

    /// Returns the length of the text covered by this token.
    #[inline]
    pub fn text_len(&self) -> TextSize {
        TextSize::of(self.text())
    }
}
