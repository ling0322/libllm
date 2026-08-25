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

//! What the scheduler needs of a model, whichever family it belongs to.

use crate::flint::{Device, Tensor};

use crate::bpe::BpeModel;
use crate::error::Result;
use crate::forward_batch::PreparedBatch;
use crate::kv_cache::{KVCacheManager, KVCacheSpec};
use crate::prompt::{Message, Prompt};
use crate::tokenizer::Tokenizer;

/// A model that can be generated from.
///
/// The scheduler drives models through this and nothing else, so a new model family only has to
/// answer these questions to be servable.
pub trait ModelForGeneration {
    /// The logits of the next token of every sequence in `batch`, `<float>(numSequences, V)`.
    /// Writes the keys and values of the batch's tokens into `cache` as it goes.
    fn forward(&self, batch: &PreparedBatch, cache: &mut KVCacheManager) -> Result<Tensor>;

    /// Whether generation is over once the model has produced this token.
    fn is_stop_token(&self, token_id: i32) -> bool;

    /// The name of the model, as its package calls it.
    fn name(&self) -> &str;

    /// Where the weights live. Everything handed to [`ModelForGeneration::forward`] belongs here.
    fn device(&self) -> Device;

    /// The width of the logits, which is the vocabulary size for most models.
    fn output_dim(&self) -> i32;

    /// The cache layout this model needs.
    fn kv_cache_spec(&self) -> Result<KVCacheSpec>;

    /// Lay a conversation out the way this model was trained to read it.
    fn build_prompt(&self, history: &[Message]) -> Result<Prompt>;

    /// The tokenizer that came with the model.
    fn tokenizer(&self) -> &Tokenizer;

    /// The vocabulary, for turning generated tokens back into text.
    fn vocab(&self) -> &BpeModel {
        self.tokenizer().vocab()
    }

    /// The tokens of a prompt.
    fn encode_prompt(&self, prompt: &Prompt) -> Result<Vec<i64>> {
        prompt.encode(self.tokenizer())
    }
}
