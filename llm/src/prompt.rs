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

//! What a model is asked to continue.
//!
//! A prompt is a run of blocks rather than one string, because the control tokens that separate a
//! conversation's turns have to reach the model as themselves: encoding `"<|eot_id|>"` as text
//! would give whatever pieces those characters happen to make, not the token the model was
//! trained on. Text blocks go through the tokenizer, control blocks are looked up by name.

use crate::bpe::BpeModel;
use crate::error::Result;
use crate::tokenizer::Tokenizer;

/// One piece of a prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptBlock {
    /// Text for the tokenizer to encode.
    Text(String),
    /// A control token, by name.
    ControlToken(String),
}

/// One turn of a conversation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A prompt, as the blocks it is built from.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Prompt {
    blocks: Vec<PromptBlock>,
}

impl Prompt {
    pub fn new() -> Prompt {
        Prompt::default()
    }

    pub fn append_text(&mut self, text: impl Into<String>) -> &mut Prompt {
        self.blocks.push(PromptBlock::Text(text.into()));
        self
    }

    pub fn append_control_token(&mut self, name: impl Into<String>) -> &mut Prompt {
        self.blocks.push(PromptBlock::ControlToken(name.into()));
        self
    }

    pub fn blocks(&self) -> &[PromptBlock] {
        &self.blocks
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// The tokens this prompt becomes.
    ///
    /// A control token that the text of a block happens to encode to is dropped: text a user
    /// typed must not be able to end its own turn or open somebody else's.
    pub fn encode(&self, tokenizer: &Tokenizer) -> Result<Vec<i64>> {
        let vocab: &BpeModel = tokenizer.vocab();
        let mut token_ids = Vec::new();

        for block in &self.blocks {
            match block {
                PromptBlock::ControlToken(name) => {
                    token_ids.push(vocab.find_control_token(name)? as i64);
                }
                PromptBlock::Text(text) => {
                    for token_id in tokenizer.encode(text) {
                        if !vocab.is_control_token(token_id)? {
                            token_ids.push(token_id as i64);
                        }
                    }
                }
            }
        }

        Ok(token_ids)
    }

    /// How many tokens this prompt becomes.
    pub fn token_count(&self, tokenizer: &Tokenizer) -> Result<usize> {
        Ok(self.encode(tokenizer)?.len())
    }
}
