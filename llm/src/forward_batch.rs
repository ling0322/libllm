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

//! How a packed batch of sequences is laid out for one forward pass.
//!
//! Several sequences are forwarded at once by packing their tokens back to back into one
//! `<float>(totalTokens, hidden)` activation. What tells them apart afterwards is the cumulative
//! sequence lengths: `cu_seqlens_q` of `[0, 3, 7]` says that the first sequence owns tokens
//! `[0, 3)` and the second owns `[3, 7)`, the convention the attention kernels take.
//!
//! A batch is built as a layout, gets its cache blocks assigned, and is then prepared for a
//! device, which turns the layout into the tensors the kernels read. Preparing gives a
//! [`PreparedBatch`], so a batch that has not been prepared cannot be handed to a model by
//! mistake, and neither can one that is still missing its blocks.

use crate::flint::{Device, Tensor};

use crate::error::{Error, Result};

/// The tokens of several sequences, packed, and where each one's keys and values belong.
#[derive(Clone, Debug, Default)]
pub struct ForwardBatch {
    token_ids: Vec<i64>,
    cu_seqlens_q: Vec<i32>,
    cu_seqlens_k: Vec<i32>,
    position_ids: Vec<i64>,
    block_ids: Vec<Vec<i32>>,
}

impl ForwardBatch {
    /// One sequence of `q_len` tokens whose cache already holds `past_len` of them, described
    /// without the tokens themselves. For measuring what a pass of a given size costs.
    pub fn single_layout(q_len: i32, past_len: i32) -> Result<ForwardBatch> {
        if q_len <= 0 || past_len < 0 {
            return Err(Error::model(format!(
                "a sequence needs at least one token and cannot have a negative past, got \
                 q_len={q_len} past_len={past_len}"
            )));
        }

        Ok(ForwardBatch {
            token_ids: Vec::new(),
            cu_seqlens_q: vec![0, q_len],
            cu_seqlens_k: vec![0, past_len + q_len],
            position_ids: (past_len..past_len + q_len).map(i64::from).collect(),
            block_ids: Vec::new(),
        })
    }

    /// One sequence, carrying the tokens to forward. This is what a scheduler hands to a model
    /// when it runs a single request.
    pub fn single(token_ids: &[i64], past_len: i32) -> Result<ForwardBatch> {
        if token_ids.is_empty() {
            return Err(Error::model("a batch needs at least one token"));
        }

        let mut batch = ForwardBatch::single_layout(token_ids.len() as i32, past_len)?;
        batch.token_ids = token_ids.to_vec();
        Ok(batch)
    }

    /// Several sequences, their tokens packed back to back.
    ///
    /// `token_ids` and `position_ids` are as long as the last entry of `cu_seqlens_q`, in packed
    /// order; `cu_seqlens_k` counts the cached tokens each sequence attends to, its own included.
    pub fn packed(
        token_ids: Vec<i64>,
        cu_seqlens_q: Vec<i32>,
        cu_seqlens_k: Vec<i32>,
        position_ids: Vec<i64>,
    ) -> Result<ForwardBatch> {
        if cu_seqlens_q.len() < 2 || cu_seqlens_q.len() != cu_seqlens_k.len() {
            return Err(Error::model(
                "a packed batch needs one cumulative length per sequence plus a leading zero, in \
                 both q and k",
            ));
        }

        let total = *cu_seqlens_q.last().expect("checked above") as usize;
        if token_ids.len() != total || position_ids.len() != total {
            return Err(Error::model(format!(
                "the cumulative lengths call for {total} packed tokens, got {} tokens and {} \
                 positions",
                token_ids.len(),
                position_ids.len()
            )));
        }

        Ok(ForwardBatch {
            token_ids,
            cu_seqlens_q,
            cu_seqlens_k,
            position_ids,
            block_ids: Vec::new(),
        })
    }

    /// Assign the blocks each sequence owns, in token order.
    pub fn set_block_ids(&mut self, block_ids: Vec<Vec<i32>>) -> Result<()> {
        if block_ids.len() != self.num_sequences() as usize {
            return Err(Error::model(format!(
                "batch holds {} sequences but was given blocks for {}",
                self.num_sequences(),
                block_ids.len()
            )));
        }

        self.block_ids = block_ids;
        Ok(())
    }

    pub fn num_sequences(&self) -> i32 {
        self.cu_seqlens_q.len() as i32 - 1
    }

    /// The packed query tokens over all sequences.
    pub fn total_q_len(&self) -> i32 {
        *self.cu_seqlens_q.last().unwrap_or(&0)
    }

    /// The keys and values all sequences attend to.
    pub fn total_k_len(&self) -> i32 {
        *self.cu_seqlens_k.last().unwrap_or(&0)
    }

    /// The longest query in the batch, which sizes the attention kernel's work.
    pub fn max_q_len(&self) -> i32 {
        self.lengths(&self.cu_seqlens_q).max().unwrap_or(0)
    }

    /// The longest key range in the batch.
    pub fn max_k_len(&self) -> i32 {
        self.lengths(&self.cu_seqlens_k).max().unwrap_or(0)
    }

    pub fn token_ids(&self) -> &[i64] {
        &self.token_ids
    }

    pub fn cu_seqlens_q(&self) -> &[i32] {
        &self.cu_seqlens_q
    }

    pub fn cu_seqlens_k(&self) -> &[i32] {
        &self.cu_seqlens_k
    }

    pub fn position_ids(&self) -> &[i64] {
        &self.position_ids
    }

    pub fn block_ids(&self) -> &[Vec<i32>] {
        &self.block_ids
    }

    /// Turn the layout into the tensors a model reads, on `device`.
    ///
    /// `block_size` is the one the KV cache was built with: it is what turns a token's position
    /// into the block that holds it and the offset inside that block.
    pub fn prepare(self, device: Device, block_size: i32) -> Result<PreparedBatch> {
        PreparedBatch::new(self, device, block_size)
    }

    fn lengths<'a>(&'a self, cumulative: &'a [i32]) -> impl Iterator<Item = i32> + 'a {
        cumulative.windows(2).map(|pair| pair[1] - pair[0])
    }
}

/// A batch with its device tensors materialized, ready to forward.
///
/// Holding these on the batch rather than rebuilding them per layer matters: a model reads the
/// same block table and slot mapping once per layer, and they do not change between layers.
#[derive(Debug)]
pub struct PreparedBatch {
    batch: ForwardBatch,
    device: Device,
    token_ids: Option<Tensor>,
    position_ids: Tensor,
    last_query_indices: Tensor,
    cu_seqlens_q: Tensor,
    seqlens_k: Tensor,
    block_table: Tensor,
    slot_mapping: Tensor,
}

impl PreparedBatch {
    fn new(batch: ForwardBatch, device: Device, block_size: i32) -> Result<PreparedBatch> {
        if batch.block_ids.is_empty() {
            return Err(Error::model("batch has no cache blocks assigned"));
        }
        if batch.position_ids.len() != batch.total_q_len() as usize {
            return Err(Error::model(
                "batch has one position per packed token, and does not have that many",
            ));
        }
        if block_size <= 0 {
            return Err(Error::model("kv cache block size must be positive"));
        }

        let to_device = |tensor: Tensor| -> Result<Tensor> { Ok(tensor.to_device(device)?) };

        let token_ids = if batch.token_ids.is_empty() {
            None
        } else {
            if batch.token_ids.len() != batch.total_q_len() as usize {
                return Err(Error::model(
                    "batch carries tokens, and does not carry one per packed position",
                ));
            }
            Some(to_device(Tensor::from_i64(
                &[batch.token_ids.len() as i32],
                &batch.token_ids,
            )?)?)
        };

        let num_sequences = batch.num_sequences();
        let mut seqlens_k = Vec::with_capacity(num_sequences as usize);
        let mut last_query_indices = Vec::with_capacity(num_sequences as usize);
        for i in 0..num_sequences as usize {
            seqlens_k.push(batch.cu_seqlens_k[i + 1] - batch.cu_seqlens_k[i]);
            // The logits that matter are the ones of each sequence's last query token.
            last_query_indices.push((batch.cu_seqlens_q[i + 1] - 1) as i64);
        }

        let max_num_blocks = batch
            .block_ids
            .iter()
            .map(Vec::len)
            .max()
            .expect("block ids are not empty");
        if max_num_blocks == 0 {
            return Err(Error::model("a sequence in the batch was given no blocks"));
        }

        // Rows are padded to the longest one. A kernel never reads past a sequence's own length,
        // so what the padding holds does not matter.
        let mut table = vec![0i32; num_sequences as usize * max_num_blocks];
        for (i, blocks) in batch.block_ids.iter().enumerate() {
            table[i * max_num_blocks..i * max_num_blocks + blocks.len()].copy_from_slice(blocks);
        }

        let mut slots = vec![0i32; batch.total_q_len() as usize];
        for i in 0..num_sequences as usize {
            let blocks = &batch.block_ids[i];
            for t in batch.cu_seqlens_q[i]..batch.cu_seqlens_q[i + 1] {
                // A token's rotary position is also its index in its sequence, so it names the
                // slot its key and value belong in.
                let position = batch.position_ids[t as usize] as i32;
                let block = (position / block_size) as usize;
                let block_id = blocks.get(block).ok_or_else(|| {
                    Error::model(format!(
                        "sequence {i} reached position {position}, past the {} blocks it owns",
                        blocks.len()
                    ))
                })?;
                slots[t as usize] = block_id * block_size + position % block_size;
            }
        }

        Ok(PreparedBatch {
            device,
            token_ids,
            position_ids: to_device(Tensor::from_i64(
                &[batch.position_ids.len() as i32],
                &batch.position_ids,
            )?)?,
            last_query_indices: to_device(Tensor::from_i64(
                &[num_sequences],
                &last_query_indices,
            )?)?,
            cu_seqlens_q: to_device(Tensor::from_i32(
                &[batch.cu_seqlens_q.len() as i32],
                &batch.cu_seqlens_q,
            )?)?,
            seqlens_k: to_device(Tensor::from_i32(&[num_sequences], &seqlens_k)?)?,
            block_table: to_device(Tensor::from_i32(
                &[num_sequences, max_num_blocks as i32],
                &table,
            )?)?,
            slot_mapping: to_device(Tensor::from_i32(&[batch.total_q_len()], &slots)?)?,
            batch,
        })
    }

    /// The layout this was prepared from.
    pub fn layout(&self) -> &ForwardBatch {
        &self.batch
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The packed tokens, `<long>(totalQLen)`. Absent when the batch only describes a layout.
    pub fn token_ids(&self) -> Option<&Tensor> {
        self.token_ids.as_ref()
    }

    /// The rotary position of every packed token, `<long>(totalQLen)`.
    pub fn position_ids(&self) -> &Tensor {
        &self.position_ids
    }

    /// Where each sequence's last query sits in the packed order, `<long>(numSequences)`. These
    /// are the rows a model takes its next-token logits from.
    pub fn last_query_indices(&self) -> &Tensor {
        &self.last_query_indices
    }

    /// `<int>(numSequences + 1)`: where each sequence starts in the packed order.
    pub fn cu_seqlens_q(&self) -> &Tensor {
        &self.cu_seqlens_q
    }

    /// `<int>(numSequences)`: how many cached tokens each sequence attends to.
    pub fn seqlens_k(&self) -> &Tensor {
        &self.seqlens_k
    }

    /// `<int>(numSequences, maxNumBlocks)`: the blocks each sequence owns, in token order.
    pub fn block_table(&self) -> &Tensor {
        &self.block_table
    }

    /// `<int>(totalQLen)`: the slot each packed token's key and value belong in, as
    /// `blockId * blockSize + offset`.
    pub fn slot_mapping(&self) -> &Tensor {
        &self.slot_mapping
    }

    pub fn num_sequences(&self) -> i32 {
        self.batch.num_sequences()
    }

    pub fn total_q_len(&self) -> i32 {
        self.batch.total_q_len()
    }

    pub fn max_q_len(&self) -> i32 {
        self.batch.max_q_len()
    }

    pub fn max_k_len(&self) -> i32 {
        self.batch.max_k_len()
    }
}
