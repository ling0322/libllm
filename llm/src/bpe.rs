// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! The byte pair encoding tokenizer.
//!
//! Encoding starts with one symbol per character and repeatedly merges the adjacent pair that the
//! model scores best, which is the usual sentencepiece algorithm. The symbols are held as a linked
//! list so that a merge is a constant-time splice, and the candidate merges sit in a heap keyed by
//! cost; merging a pair invalidates the two symbols it consumed, so a candidate is checked for
//! still being alive when it comes off the heap rather than being removed when it dies.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::io::Read;

use crate::error::{Error, Result};
use crate::ini::IniSection;
use crate::reader::BinaryRead;

/// The id that stands for no token at all.
pub const INVALID_TOKEN: i32 = -1;

/// What a token is, beyond a piece of text. A token with no flags is an ordinary one.
mod flags {
    pub const UNKNOWN: i8 = 1;
    pub const CONTROL: i8 = 2;
    pub const BYTE: i8 = 4;
    #[allow(dead_code)]
    pub const UNUSED: i8 = 8;
}

/// How a model wants its text prepared before it is encoded.
#[derive(Clone, Debug)]
pub struct BpeConfig {
    /// The parameter file holding the vocabulary.
    pub model_file: String,
    /// Whether to put a space in front of the text, which is what makes the first word look like
    /// any other word to a model trained on space-prefixed pieces.
    pub add_prefix_space: bool,
    /// Whether the initial symbols are characters rather than bytes.
    pub split_by_unicode: bool,
}

impl Default for BpeConfig {
    fn default() -> BpeConfig {
        BpeConfig {
            model_file: String::new(),
            add_prefix_space: true,
            split_by_unicode: true,
        }
    }
}

impl BpeConfig {
    pub fn from_section(section: &IniSection) -> Result<BpeConfig> {
        Ok(BpeConfig {
            model_file: section.get_str("model_file")?.to_string(),
            add_prefix_space: section.get_bool("add_prefix_space")?,
            split_by_unicode: section.get_bool("split_by_unicode")?,
        })
    }
}

/// One entry of the vocabulary.
#[derive(Clone, Debug)]
struct TokenInfo {
    id: i32,
    weight: f32,
    /// The bytes this token stands for, which is what merging concatenates. Not text: a byte
    /// fallback token holds one raw byte, and only a whole sequence of tokens is expected to come
    /// out as valid UTF-8.
    piece: Vec<u8>,
    /// How the token is written for a person to read, and how a control token is named.
    string: Vec<u8>,
    flag: i8,
}

impl TokenInfo {
    fn is_special(&self) -> bool {
        self.flag != 0
    }
}

/// The vocabulary of a BPE tokenizer, read from the package.
#[derive(Debug)]
pub struct BpeModel {
    tokens: Vec<TokenInfo>,
    /// The ordinary tokens, by the bytes they stand for.
    token_dict: HashMap<Vec<u8>, i32>,
    /// The control and unknown tokens, by name.
    control_token_dict: HashMap<Vec<u8>, i32>,
    /// The token for each single byte, for text the vocabulary has no piece for.
    byte_id: [i32; 256],
    byte_tokens_available: bool,
    unk_id: i32,
    space_id: i32,
}

impl BpeModel {
    const MAGIC: i16 = 0x55aa;

    /// Read a vocabulary. The format is the one `tokenizer_exporter.py` writes, not a
    /// sentencepiece model as it comes.
    pub fn from_reader(reader: &mut impl Read) -> Result<BpeModel> {
        reader.expect_tag("LLsp")?;
        let num_tokens = reader.read_i32()?;
        if num_tokens <= 0 {
            return Err(Error::format(format!(
                "vocabulary holds {num_tokens} tokens"
            )));
        }
        Self::expect_magic(reader)?;

        let mut tokens = Vec::with_capacity(num_tokens as usize);
        for id in 0..num_tokens {
            tokens.push(Self::read_record(reader, id)?);
        }
        Self::expect_magic(reader)?;

        Self::build(tokens)
    }

    fn expect_magic(reader: &mut impl Read) -> Result<()> {
        if reader.read_i16()? != Self::MAGIC {
            return Err(Error::format("vocabulary is not laid out as expected"));
        }
        Ok(())
    }

    fn read_record(reader: &mut impl Read, id: i32) -> Result<TokenInfo> {
        let flag = reader.read_u8()? as i8;

        let piece_length = reader.read_u8()? as usize;
        let piece = reader.read_exact_bytes(piece_length)?;
        if flag & flags::BYTE != 0 && piece.len() != 1 {
            return Err(Error::format(format!(
                "token {id} stands for a single byte but holds {piece:?}"
            )));
        }

        let string_length = reader.read_u8()? as usize;
        let string = reader.read_exact_bytes(string_length)?;

        let mut weight = [0u8; 4];
        reader.read_exact(&mut weight)?;

        Ok(TokenInfo {
            id,
            weight: f32::from_le_bytes(weight),
            piece,
            string,
            flag,
        })
    }

    /// Indexes the tokens by what they stand for, which is what encoding looks them up by.
    fn build(tokens: Vec<TokenInfo>) -> Result<BpeModel> {
        let mut token_dict = HashMap::new();
        let mut control_token_dict = HashMap::new();
        let mut byte_id = [INVALID_TOKEN; 256];
        let mut byte_tokens_available = false;
        let mut unk_id = INVALID_TOKEN;

        for info in &tokens {
            if info.flag == 0 {
                token_dict.insert(info.piece.clone(), info.id);
            } else if info.flag & flags::BYTE != 0 {
                byte_tokens_available = true;
                byte_id[info.piece[0] as usize] = info.id;
            } else if info.flag & flags::UNKNOWN != 0 {
                if unk_id != INVALID_TOKEN {
                    return Err(Error::format(
                        "vocabulary holds more than one unknown token",
                    ));
                }
                unk_id = info.id;
                control_token_dict.insert(info.string.clone(), info.id);
            } else if info.flag & flags::CONTROL != 0 {
                control_token_dict.insert(info.string.clone(), info.id);
            }
        }

        let space_id = *token_dict
            .get(b" ".as_slice())
            .ok_or_else(|| Error::format("vocabulary has no token for a space"))?;

        // Byte fallback only works if every byte has a token, so a model that offers some of them
        // and not others would fail on text nobody tested.
        if byte_tokens_available {
            if let Some(byte) = byte_id.iter().position(|&id| id == INVALID_TOKEN) {
                return Err(Error::format(format!(
                    "vocabulary has byte tokens but none for byte {byte}"
                )));
            }
        }

        Ok(BpeModel {
            tokens,
            token_dict,
            control_token_dict,
            byte_id,
            byte_tokens_available,
            unk_id,
            space_id,
        })
    }

    /// The id of `piece`, or the unknown token when the vocabulary has no such piece.
    pub fn find_token(&self, piece: &[u8]) -> i32 {
        self.token_dict.get(piece).copied().unwrap_or(self.unk_id)
    }

    /// The id of a control token, by name. A missing one is an error rather than the unknown
    /// token: a prompt template naming a control token the model does not have is a mistake.
    pub fn find_control_token(&self, name: &str) -> Result<i32> {
        self.control_token_dict
            .get(name.as_bytes())
            .copied()
            .ok_or_else(|| Error::model(format!("control token {name:?} is not in the vocabulary")))
    }

    /// The bytes a token stands for. Empty for a control, unknown or unused token.
    pub fn token_piece(&self, token_id: i32) -> Result<&[u8]> {
        Ok(&self.info(token_id)?.piece)
    }

    /// How a token is written for a person to read, as bytes: one token on its own is not
    /// necessarily whole text, so joining a run of them and decoding once is what gives a string.
    pub fn token_string(&self, token_id: i32) -> Result<&[u8]> {
        Ok(&self.info(token_id)?.string)
    }

    /// The bytes a run of tokens stands for, as text.
    ///
    /// This is the reconstruction of what was encoded, where [`BpeModel::decode`] gives the
    /// display form; the two differ wherever the vocabulary writes a piece differently from the
    /// bytes it stands for, such as a space or a newline.
    pub fn decode_pieces(&self, token_ids: &[i32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            bytes.extend_from_slice(self.token_piece(token_id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The text of a run of tokens, taken together.
    ///
    /// Decoding once at the end rather than token by token is what makes byte fallback work: a
    /// character split across several tokens is only valid UTF-8 once they are back together.
    pub fn decode(&self, token_ids: &[i32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            bytes.extend_from_slice(self.token_string(token_id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn vocab_size(&self) -> i32 {
        self.tokens.len() as i32
    }

    /// Whether a token stands for something other than text, which is what keeps it from being
    /// merged into its neighbours.
    pub fn is_control_token(&self, token_id: i32) -> Result<bool> {
        Ok(self.info(token_id)?.is_special())
    }

    /// The unknown token, or [`INVALID_TOKEN`] when the vocabulary has none.
    pub fn unk_id(&self) -> i32 {
        self.unk_id
    }

    pub fn space_id(&self) -> i32 {
        self.space_id
    }

    pub fn byte_tokens_available(&self) -> bool {
        self.byte_tokens_available
    }

    /// The token standing for one byte.
    pub fn byte_id(&self, byte: u8) -> i32 {
        self.byte_id[byte as usize]
    }

    /// The token that `left` and `right` merge into, and what that merge costs. Costs are negated
    /// weights, so the best merge is the cheapest one.
    fn find_merge(&self, left: i32, right: i32) -> Option<(i32, f32)> {
        let mut merged = self.info(left).ok()?.piece.clone();
        merged.extend_from_slice(&self.info(right).ok()?.piece);
        let id = *self.token_dict.get(&merged)?;
        Some((id, -self.info(id).ok()?.weight))
    }

    fn info(&self, token_id: i32) -> Result<&TokenInfo> {
        if token_id < 0 {
            return Err(Error::model(format!("{token_id} is not a token id")));
        }
        self.tokens
            .get(token_id as usize)
            .ok_or_else(|| Error::model(format!("token {token_id} is past the vocabulary")))
    }

    fn is_special(&self, token_id: i32) -> bool {
        self.info(token_id)
            .map(TokenInfo::is_special)
            .unwrap_or(true)
    }
}

/// One symbol of the list being merged. Symbols live in an arena and refer to each other by
/// index, which is the same linked list the C++ builds out of pointers.
#[derive(Clone, Copy, Debug)]
struct Symbol {
    prev: Option<usize>,
    next: Option<usize>,
    token_id: i32,
}

impl Symbol {
    /// A symbol that a merge consumed is left behind in the arena, and skipped from then on.
    fn valid(&self) -> bool {
        self.token_id != INVALID_TOKEN
    }
}

/// A merge waiting to happen.
#[derive(Clone, Copy, Debug)]
struct Bigram {
    left: usize,
    right: usize,
    cost: f32,
    merged_token_id: i32,
}

impl Ord for Bigram {
    /// Reversed, so that the [`BinaryHeap`] gives up the cheapest merge first. Equal costs go to
    /// the leftmost pair, which keeps the encoding of a given string from depending on the order
    /// candidates happened to be pushed.
    fn cmp(&self, other: &Bigram) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Bigram) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Bigram) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Bigram {}

/// Encodes text with one vocabulary.
pub struct BpeEncoder<'a> {
    model: &'a BpeModel,
    config: &'a BpeConfig,
    symbols: Vec<Symbol>,
    queue: BinaryHeap<Bigram>,
    /// The list runs from a header symbol that is never merged, so that a merge always has a
    /// previous symbol to splice onto.
    header: usize,
}

impl<'a> BpeEncoder<'a> {
    pub fn new(model: &'a BpeModel, config: &'a BpeConfig) -> BpeEncoder<'a> {
        BpeEncoder {
            model,
            config,
            symbols: Vec::new(),
            queue: BinaryHeap::new(),
            header: 0,
        }
    }

    /// The token ids of `text`.
    pub fn encode(&mut self, text: &str) -> Vec<i32> {
        self.init_symbol_list(text);
        self.init_queue();

        while let Some(bigram) = self.queue.pop() {
            // Either side may have been merged into something else since this was queued.
            if self.symbols[bigram.left].valid() && self.symbols[bigram.right].valid() {
                let merged = self.merge_bigram(&bigram);
                self.add_bigram_if_exists(self.symbols[merged].prev, Some(merged));
                self.add_bigram_if_exists(Some(merged), self.symbols[merged].next);
            }
        }

        self.token_ids()
    }

    fn init_symbol_list(&mut self, text: &str) {
        self.symbols.clear();
        self.queue.clear();

        self.header = self.alloc(INVALID_TOKEN);
        self.symbols[self.header].prev = None;
        let mut tail = self.header;

        if self.config.add_prefix_space {
            tail = self.append_token(tail, self.model.space_id());
        }

        // A piece the vocabulary has no token for falls back to one token per byte.
        for piece in self.split(text) {
            let token_id = if piece == b" " {
                self.model.space_id()
            } else {
                self.model.find_token(&piece)
            };

            if token_id == self.model.unk_id() && self.model.byte_tokens_available() {
                for &byte in &piece {
                    tail = self.append_token(tail, self.model.byte_id(byte));
                }
            } else {
                tail = self.append_token(tail, token_id);
            }
        }
    }

    /// The initial symbols: characters, or bytes for a model that was trained on them.
    fn split(&self, text: &str) -> Vec<Vec<u8>> {
        if self.config.split_by_unicode {
            text.chars()
                .map(|character| character.to_string().into_bytes())
                .collect()
        } else {
            text.bytes().map(|byte| vec![byte]).collect()
        }
    }

    fn init_queue(&mut self) {
        let mut left = self.symbols[self.header].next;
        while let Some(index) = left {
            let right = self.symbols[index].next;
            self.add_bigram_if_exists(Some(index), right);
            left = right;
        }
    }

    fn add_bigram_if_exists(&mut self, left: Option<usize>, right: Option<usize>) {
        let (Some(left), Some(right)) = (left, right) else {
            return;
        };
        // The header is not a token, and a control token never merges with its neighbours.
        if left == self.header
            || self.model.is_special(self.symbols[left].token_id)
            || self.model.is_special(self.symbols[right].token_id)
        {
            return;
        }

        if let Some((merged_token_id, cost)) = self
            .model
            .find_merge(self.symbols[left].token_id, self.symbols[right].token_id)
        {
            self.queue.push(Bigram {
                left,
                right,
                cost,
                merged_token_id,
            });
        }
    }

    /// Splices the merged symbol in and marks the two it replaced as gone.
    fn merge_bigram(&mut self, bigram: &Bigram) -> usize {
        let prev = self.symbols[bigram.left].prev;
        let next = self.symbols[bigram.right].next;

        let merged = self.alloc(bigram.merged_token_id);
        self.symbols[merged].prev = prev;
        self.symbols[merged].next = next;

        if let Some(next) = next {
            self.symbols[next].prev = Some(merged);
        }
        // There is always a previous symbol, since the list starts with the header.
        if let Some(prev) = prev {
            self.symbols[prev].next = Some(merged);
        }

        for consumed in [bigram.left, bigram.right] {
            self.symbols[consumed].token_id = INVALID_TOKEN;
            self.symbols[consumed].prev = None;
            self.symbols[consumed].next = None;
        }

        merged
    }

    fn append_token(&mut self, tail: usize, token_id: i32) -> usize {
        let symbol = self.alloc(token_id);
        self.symbols[symbol].prev = Some(tail);
        self.symbols[tail].next = Some(symbol);
        symbol
    }

    fn alloc(&mut self, token_id: i32) -> usize {
        self.symbols.push(Symbol {
            prev: None,
            next: None,
            token_id,
        });
        self.symbols.len() - 1
    }

    fn token_ids(&self) -> Vec<i32> {
        let mut ids = Vec::new();
        let mut current = self.symbols[self.header].next;
        while let Some(index) = current {
            ids.push(self.symbols[index].token_id);
            current = self.symbols[index].next;
        }
        ids
    }
}
