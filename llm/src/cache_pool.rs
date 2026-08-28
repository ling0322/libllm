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

//! One pool of bytes, which every layer of a model draws its cache from.
//!
//! The pool is a fixed number of **blocks** of a fixed size, held as one tensor of raw bytes per
//! layer. A block id names row `b` — one **page** — of every one of them at once, so a sequence
//! carries one list of ids rather than one per layer, and — the point of holding it this way — a
//! block is the same bytes whoever takes it. What a layer keeps in a block it took is its own
//! business: a full attention layer reads it as the keys and values of a fixed number of tokens, a
//! gated DeltaNet layer reads it as part of a recurrent state, and neither can be handed a block
//! the other holds.
//! That is what one pool buys over a pool per kind: the memory a model does not spend on state is
//! context, and it does not have to be divided up in advance.
//!
//! # Groups and slots
//!
//! The layers of the model are divided into **groups** of equal size, the way vLLM's hybrid
//! allocator divides them: layers that need the same thing of a block are one group, and a kind
//! with more layers than the smallest one is split into several groups rather than given bigger
//! blocks. A Qwen3.5 stack has 16 full attention layers and 48 gated DeltaNet ones, so it has four
//! groups of 16 — one attention, three recurrent.
//!
//! A **slot** is a position within a group, and it is what a byte tensor belongs to: slot `i` is
//! shared by layer `i` of every group, one page of it read as whatever the layer that took that
//! block keeps there. Nothing keeps two layers apart except the block ids they hold, and nothing
//! has to: a block belongs to one group at a time, so its bytes are read as one thing at a time.
//!
//! # Blocks and runs
//!
//! Two sizes come out of the pool. The base one is a block, which is what a group of attention
//! layers takes one of as its sequence grows. The large one is a **run** of `blocks_per_run`
//! *consecutive* blocks, which is what a group of recurrent layers takes one of per request: a
//! state is one fixed-size thing however long the sequence gets, and it is far larger than a block
//! — 3.2 MB against 64 KB on the 27B — so it spans a run and is indexed by the run's number.
//!
//! Runs are why the blocks stay small. vLLM makes one state one block by raising the block size
//! until a block holds a state, which on that model is 784 tokens of attention a block and about
//! 25 MB of context wasted per request. A run costs the padding at its end instead, which is under
//! a block.
//!
//! What the two sizes must not do is make each other unallocatable, and they are not symmetric
//! about it: a request that cannot get another block can be preempted and resumed, where one that
//! cannot get its state cannot run at all. So a floor of whole runs is kept back from the base
//! size, and only the runs above that floor may be broken up.
//!
//! # The allocator
//!
//! The pool is divided into runs of `blocks_per_run` blocks, and each run carries a bitmap of the
//! blocks in it that are taken. A run is *free* (its bitmap is zero), *full* (all ones), or
//! *partial*. Free runs are on one stack and partial ones on another, and every operation is a
//! handful of instructions on the top of a stack:
//!
//! - taking a block reads the run on top of the partial stack and takes its lowest free bit, which
//!   is one `trailing_zeros`; it breaks a free run open only when there is no partial one.
//! - taking a run pops the free stack.
//! - giving either back pushes a stack, and moves the run between stacks when its bitmap becomes
//!   all ones or all zeros.
//!
//! The partial stack is allowed to hold runs that are no longer partial rather than paying to
//! remove them from the middle: a run that has since filled up or emptied out is recognised by its
//! bitmap and dropped when it reaches the top. Nothing is searched, nothing is coalesced, and
//! nothing allocates.
//!
//! Reading the partial stack from the top also keeps the fragmentation down on its own: blocks come
//! out of the same run until it is full, so the runs that have not been broken into stay whole and
//! there is something left for the large size to take.

use crate::flint::{functional as F, DType, Device, Tensor};

use crate::error::{Error, Result};

/// The most blocks a run may hold, which is what a run's bitmap fits in.
pub const MAX_BLOCKS_PER_RUN: i32 = 64;

/// What every homogeneous page starts on, so that an element never straddles the boundary a byte
/// view puts it at. Eight rather than four because it is the widest element a page can hold.
const ALIGNMENT: i64 = 8;

/// The bytes a pool is made of: what one layer keeps at one block id, how many layers share a
/// block, and how many of them the pool holds.
///
/// A **page** is one layer's share of one block — `page_size_bytes` bytes of the byte tensor that
/// layer reads. A **block** is page `b` of every one of the `num_layers` layers at once, which is
/// what lets one id name it in all of them. The pool is `num_pages` pages deep, so it holds that
/// many blocks and `page_size_bytes * num_layers * num_pages` bytes in all.
///
/// It says nothing about what a page holds. That is decided by the group that took the block, in
/// [`GroupShape`], and a page is large enough for any of them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockShape {
    /// What one layer keeps at one block id.
    pub page_size_bytes: i64,
    /// The layers that share a block, which is one page each. Every group has this many, so it is
    /// also how many byte tensors the pool holds.
    pub num_layers: i32,
    /// How many pages deep the pool is, which is how many blocks it holds.
    pub num_pages: i32,
}

impl BlockShape {
    pub fn new(page_size_bytes: i64, num_layers: i32, num_pages: i32) -> Result<BlockShape> {
        let shape = BlockShape {
            page_size_bytes,
            num_layers,
            num_pages,
        };
        shape.validate()?;
        Ok(shape)
    }

    /// Whether the pool it describes can be built at all.
    pub fn validate(&self) -> Result<()> {
        let BlockShape {
            page_size_bytes,
            num_layers,
            num_pages,
        } = *self;
        if page_size_bytes <= 0 {
            return Err(Error::model(format!(
                "a page has to be at least one byte, got {page_size_bytes}"
            )));
        }
        if num_layers <= 0 {
            return Err(Error::model(format!(
                "a pool needs at least one layer per group, got {num_layers}"
            )));
        }
        if num_pages <= 0 {
            return Err(Error::model(format!(
                "a pool needs at least one page, got {num_pages}"
            )));
        }
        if page_size_bytes * i64::from(num_pages) > i64::from(i32::MAX) {
            return Err(Error::model(format!(
                "a layer of this pool holds {} bytes, which does not fit in a tensor dimension",
                page_size_bytes * i64::from(num_pages)
            )));
        }

        Ok(())
    }

    /// The bytes one block takes across every layer, which is what one block id costs.
    pub fn bytes_per_block(&self) -> i64 {
        self.page_size_bytes * i64::from(self.num_layers)
    }

    /// Everything the pool holds.
    pub fn bytes(&self) -> i64 {
        self.bytes_per_block() * i64::from(self.num_pages)
    }
}

/// How much of the pool one allocation of a group takes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheUnit {
    /// One block, taken again every time the sequence outgrows the last one.
    Block,
    /// One run of consecutive blocks, taken once and held until the request is done.
    Run,
}

/// One group of layers: what each of them keeps in an allocation, and how big an allocation is.
///
/// What a layer keeps is logical tensor shapes and their common element type, without the block
/// dimension the pool puts in front. They are consecutive regions of one homogeneous array, in
/// this order, laid over the page a group that takes blocks is given or over the whole run a group
/// that takes runs is given.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupShape {
    pub shapes: Vec<Vec<i32>>,
    pub dtype: DType,
    pub unit: CacheUnit,
}

impl GroupShape {
    pub fn new(shapes: Vec<Vec<i32>>, dtype: DType, unit: CacheUnit) -> Result<GroupShape> {
        if shapes.is_empty() {
            return Err(Error::model("a group has to carry at least one tensor"));
        }
        for shape in &shapes {
            if shape.is_empty() || shape.iter().any(|&d| d <= 0) {
                return Err(Error::model(format!(
                    "a group's shape must be positive in every dimension, got {shape:?}"
                )));
            }
        }

        Ok(GroupShape {
            shapes,
            dtype,
            unit,
        })
    }

    /// Where each tensor starts inside an allocation, and the bytes the whole homogeneous array
    /// takes. The tensors are consecutive elements of [`GroupShape::dtype`]; only the end is
    /// rounded up, so that what follows starts on an [`ALIGNMENT`] boundary.
    pub fn layout(&self) -> (Vec<i64>, i64) {
        let mut offsets = Vec::with_capacity(self.shapes.len());
        let mut at = 0i64;
        for shape in &self.shapes {
            offsets.push(at);
            let numel = shape.iter().map(|&d| i64::from(d)).product();
            at += self.dtype.total_size(numel);
        }
        let bytes = (at + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        (offsets, bytes)
    }

    /// The bytes one layer of this group keeps in one allocation, counting what the alignment
    /// adds.
    pub fn bytes(&self) -> i64 {
        self.layout().1
    }
}

/// Which stack a run is on, which is decided by its bitmap rather than stored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunState {
    Free,
    Partial,
    Full,
}

/// The allocator described in the module comment. It knows nothing about tensors: it hands out
/// block ids and run numbers.
#[derive(Debug)]
struct RunAllocator {
    blocks_per_run: i32,
    /// All ones for the blocks a run holds, which is what "full" compares against.
    full_mask: u64,
    /// The blocks taken in each run.
    used: Vec<u64>,
    /// Runs whose bitmap is zero. No duplicates: a run is pushed here only when it empties, and it
    /// cannot empty twice without being allocated from in between.
    free_runs: Vec<i32>,
    /// Runs that had a free block and a taken one when they were pushed. May hold runs that are no
    /// longer either, which is what makes pushing free; the bitmap says which.
    partial: Vec<i32>,
    /// Whether a run is already on the partial stack, so it is not pushed twice.
    on_partial: Vec<bool>,
    /// Whole runs that the base size may not break into, counting the ones already handed out.
    /// Keeps a burst of block allocations from leaving nothing for the size that cannot be
    /// preempted.
    reserved_runs: i32,
    /// The runs handed out whole, which are the reservation already met.
    outstanding_runs: i32,
    num_free_blocks: i32,
}

impl RunAllocator {
    fn new(num_runs: i32, blocks_per_run: i32, reserved_runs: i32) -> RunAllocator {
        RunAllocator {
            blocks_per_run,
            full_mask: if blocks_per_run == 64 {
                u64::MAX
            } else {
                (1u64 << blocks_per_run) - 1
            },
            used: vec![0; num_runs as usize],
            // Descending, so that the lowest run is taken first: a pool that is never full then
            // keeps its traffic in the same few runs.
            free_runs: (0..num_runs).rev().collect(),
            partial: Vec::new(),
            on_partial: vec![false; num_runs as usize],
            reserved_runs,
            outstanding_runs: 0,
            num_free_blocks: num_runs * blocks_per_run,
        }
    }

    /// The runs still to be kept back: what was reserved, less what has already been taken. A
    /// request holding its run has met its own share of the reservation, so the blocks stop being
    /// held for it.
    fn runs_to_keep(&self) -> i32 {
        (self.reserved_runs - self.outstanding_runs).max(0)
    }

    fn state(&self, run: i32) -> RunState {
        let used = self.used[run as usize];
        if used == 0 {
            RunState::Free
        } else if used == self.full_mask {
            RunState::Full
        } else {
            RunState::Partial
        }
    }

    /// The run at the top of the partial stack that really is partial, dropping the ones that have
    /// filled up or emptied out since they were pushed.
    fn top_partial(&mut self) -> Option<i32> {
        while let Some(&run) = self.partial.last() {
            if self.state(run) == RunState::Partial {
                return Some(run);
            }
            self.partial.pop();
            self.on_partial[run as usize] = false;
        }
        None
    }

    /// Take one block. Breaks a run open only if there is no partial one and the floor allows it.
    fn allocate_block(&mut self) -> Option<i32> {
        let run = match self.top_partial() {
            Some(run) => run,
            None => {
                if self.free_runs.len() as i32 <= self.runs_to_keep() {
                    return None;
                }
                let run = self.free_runs.pop()?;
                self.partial.push(run);
                self.on_partial[run as usize] = true;
                run
            }
        };

        let free = !self.used[run as usize] & self.full_mask;
        let bit = free.trailing_zeros() as i32;
        self.used[run as usize] |= 1u64 << bit;
        self.num_free_blocks -= 1;
        Some(run * self.blocks_per_run + bit)
    }

    /// Take a whole run, whichever one is cheapest to give.
    fn allocate_run(&mut self) -> Option<i32> {
        let run = self.free_runs.pop()?;
        self.used[run as usize] = self.full_mask;
        self.num_free_blocks -= self.blocks_per_run;
        self.outstanding_runs += 1;
        Some(run)
    }

    fn free_block(&mut self, block: i32) {
        let run = block / self.blocks_per_run;
        let bit = block % self.blocks_per_run;
        let was = self.state(run);

        self.used[run as usize] &= !(1u64 << bit);
        self.num_free_blocks += 1;

        if self.state(run) == RunState::Free {
            // Its stale entry on the partial stack, if it has one, is dropped when it surfaces.
            self.free_runs.push(run);
        } else if was == RunState::Full && !self.on_partial[run as usize] {
            self.partial.push(run);
            self.on_partial[run as usize] = true;
        }
    }

    fn free_run(&mut self, run: i32) {
        self.num_free_blocks += (self.used[run as usize].count_ones()) as i32;
        self.used[run as usize] = 0;
        self.free_runs.push(run);
        self.outstanding_runs -= 1;
    }

    /// The blocks the base size may still take: what is free, less the floor kept for runs.
    fn num_allocatable_blocks(&self) -> i32 {
        (self.num_free_blocks - self.runs_to_keep() * self.blocks_per_run).max(0)
    }
}

/// A pool of blocks, and the storage every one of them names.
#[derive(Debug)]
pub struct CachePool {
    /// One tensor per layer, `<uint8>(numPages, pageSizeBytes)`. Held as bytes because the layers
    /// that share a page do not agree on a type: what a block is read as is decided by the group
    /// that took it, in `views`.
    storage: Vec<Tensor>,
    /// `[group][layer][tensor]`: the views the layers of a group read their allocations through,
    /// each `(numBlocks, ...)` for a group that takes blocks and `(numRuns, ...)` for one that
    /// takes runs, so the index into them is the id the allocator handed out.
    views: Vec<Vec<Vec<Tensor>>>,
    groups: Vec<GroupShape>,
    /// The bytes the pool is, with `num_pages` rounded down to the whole runs it was cut into.
    shape: BlockShape,
    blocks_per_run: i32,
    alloc: RunAllocator,
}

impl CachePool {
    /// How large a page has to be for these groups, and how many blocks of that size a run spans.
    ///
    /// A page is what the largest group that takes blocks needs of one, since a block is the same
    /// bytes whichever group takes it. A pool of nothing but runs has no page size of its own, so
    /// its page is a whole state and its run is one block.
    ///
    /// This is the smallest page that serves the groups. A caller that wants a larger one — to
    /// round it up, or to cut the run shorter — may pass one to [`CachePool::new`], which takes
    /// the page size from the [`BlockShape`] rather than working it out again.
    ///
    /// Answered without allocating anything, so that a caller can work out what a pool of a given
    /// size would hold before it builds one.
    pub fn page_layout(groups: &[GroupShape]) -> Result<(i64, i32)> {
        let page_size_bytes = groups
            .iter()
            .filter(|group| group.unit == CacheUnit::Block)
            .map(GroupShape::bytes)
            .max()
            .or_else(|| groups.iter().map(GroupShape::bytes).max())
            .ok_or_else(|| Error::model("a pool needs at least one group of layers"))?;

        let blocks_per_run = CachePool::run_length(groups, page_size_bytes)?;
        Ok((page_size_bytes, blocks_per_run))
    }

    /// How many blocks a run spans, for a pool whose pages are this large: as many as the largest
    /// group that takes runs needs, and one when no group takes them.
    ///
    /// Errors when a page is too small for a group that keeps what it has in one, which is the
    /// only way a page size the caller chose can fail to serve the groups.
    pub fn run_length(groups: &[GroupShape], page_size_bytes: i64) -> Result<i32> {
        if groups.is_empty() {
            return Err(Error::model("a pool needs at least one group of layers"));
        }
        if page_size_bytes <= 0 {
            return Err(Error::model(format!(
                "a page has to be at least one byte, got {page_size_bytes}"
            )));
        }

        let mut blocks_per_run = 1;
        for group in groups {
            let bytes = group.bytes();
            match group.unit {
                CacheUnit::Block if bytes > page_size_bytes => {
                    return Err(Error::model(format!(
                        "a layer of this group keeps {bytes} bytes in a page of {page_size_bytes}"
                    )));
                }
                CacheUnit::Block => {}
                CacheUnit::Run => {
                    let blocks = (bytes + page_size_bytes - 1) / page_size_bytes;
                    if blocks > i64::from(MAX_BLOCKS_PER_RUN) {
                        return Err(Error::model(format!(
                            "a state spans {blocks} blocks of {page_size_bytes} bytes and a run \
                             holds at most {MAX_BLOCKS_PER_RUN}; a larger page would fit it in \
                             fewer"
                        )));
                    }
                    blocks_per_run = blocks_per_run.max(blocks as i32);
                }
            }
        }

        Ok(blocks_per_run)
    }

    /// Allocate the storage of every layer and the allocator over it.
    ///
    /// `shape` is the bytes the pool is made of, and the caller chooses it: what the groups need
    /// of a page is [`CachePool::page_layout`], and turning a budget in bytes into a page count is
    /// the caller's own arithmetic. The run length is not passed in — it is how many pages of this
    /// size the largest group that takes runs needs, which has to hold for every group at once.
    ///
    /// `shape.num_pages` is rounded down to a whole number of runs.
    ///
    /// `reserved_runs` whole runs are kept back from [`CachePool::allocate_block`], for the callers
    /// that need a run to run at all. Runs already handed out count towards it.
    pub fn new(
        shape: BlockShape,
        groups: Vec<GroupShape>,
        reserved_runs: i32,
        device: Device,
    ) -> Result<CachePool> {
        shape.validate()?;
        let BlockShape {
            page_size_bytes,
            num_layers,
            num_pages,
        } = shape;
        let blocks_per_run = CachePool::run_length(&groups, page_size_bytes)?;

        // Whole runs only: the pages at the end of the pool that do not make up a run could never
        // be handed to a group that takes runs, and the bookkeeping is one bitmap per run.
        let num_runs = num_pages / blocks_per_run;
        if num_runs <= 0 {
            return Err(Error::model(format!(
                "a run spans {blocks_per_run} blocks and the pool was given {num_pages} pages"
            )));
        }
        let shape = BlockShape {
            page_size_bytes,
            num_layers,
            num_pages: num_runs * blocks_per_run,
        };
        if reserved_runs < 0 || reserved_runs > num_runs {
            return Err(Error::model(format!(
                "reserved_runs must be within 0..={num_runs}, got {reserved_runs}"
            )));
        }

        // Nothing is zeroed here. A block of keys and values is written before it is read, and a
        // run is zeroed when it is handed out, which is the only moment what it held before stops
        // being the caller's own.
        let mut storage = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            storage.push(Tensor::empty(
                &[shape.num_pages, page_size_bytes as i32],
                DType::UInt8,
                device,
            )?);
        }

        let mut views = Vec::with_capacity(groups.len());
        for group in &groups {
            let (offsets, _) = group.layout();
            let (rows, row_bytes) = match group.unit {
                CacheUnit::Block => (shape.num_pages, page_size_bytes),
                CacheUnit::Run => (num_runs, page_size_bytes * i64::from(blocks_per_run)),
            };

            let mut group_views = Vec::with_capacity(num_layers as usize);
            for layer in &storage {
                let rows_of_layer = layer.view(&[rows, row_bytes as i32])?;
                let mut tensors = Vec::with_capacity(group.shapes.len());
                for (i, tensor_shape) in group.shapes.iter().enumerate() {
                    let dtype = group.dtype;
                    let numel: i64 = tensor_shape.iter().map(|&d| i64::from(d)).product();
                    let begin = offsets[i];
                    let end = begin + dtype.total_size(numel);
                    let mut view_shape = Vec::with_capacity(tensor_shape.len() + 1);
                    view_shape.push(rows);
                    view_shape.extend_from_slice(tensor_shape);
                    tensors.push(
                        rows_of_layer
                            .slice(1, begin as i32, end as i32)?
                            .view_as(dtype)?
                            .view(&view_shape)?,
                    );
                }
                group_views.push(tensors);
            }
            views.push(group_views);
        }

        Ok(CachePool {
            storage,
            views,
            groups,
            shape,
            blocks_per_run,
            alloc: RunAllocator::new(num_runs, blocks_per_run, reserved_runs),
        })
    }

    /// The tensors one layer reads its allocations through: `slot` of `group`, each in the order
    /// the group shape gave them, and each indexed by the id the allocator handed out.
    pub fn layer(&self, group: i32, slot: i32) -> Result<&[Tensor]> {
        self.views
            .get(group as usize)
            .and_then(|group| group.get(slot as usize))
            .map(Vec::as_slice)
            .ok_or_else(|| Error::model(format!("slot {slot} of group {group} is not in this pool")))
    }

    pub fn layer_mut(&mut self, group: i32, slot: i32) -> Result<&mut [Tensor]> {
        self.views
            .get_mut(group as usize)
            .and_then(|group| group.get_mut(slot as usize))
            .map(Vec::as_mut_slice)
            .ok_or_else(|| Error::model(format!("slot {slot} of group {group} is not in this pool")))
    }

    pub fn num_groups(&self) -> i32 {
        self.views.len() as i32
    }

    /// The bytes the pool is made of, with the pages that did not make up a whole run taken off.
    pub fn shape(&self) -> BlockShape {
        self.shape
    }

    /// The layers each group has, which is how many tensors of bytes the pool holds.
    pub fn num_layers(&self) -> i32 {
        self.shape.num_layers
    }

    pub fn num_blocks(&self) -> i32 {
        self.shape.num_pages
    }

    pub fn num_runs(&self) -> i32 {
        self.shape.num_pages / self.blocks_per_run
    }

    pub fn blocks_per_run(&self) -> i32 {
        self.blocks_per_run
    }

    /// The bytes one block takes across every layer of the pool, which is what one block id costs.
    pub fn bytes_per_block(&self) -> i64 {
        self.shape.bytes_per_block()
    }

    /// Everything the pool holds.
    pub fn bytes(&self) -> i64 {
        self.shape.bytes()
    }

    pub fn group(&self, group: i32) -> Result<&GroupShape> {
        self.groups
            .get(group as usize)
            .ok_or_else(|| Error::model(format!("group {group} is not in this pool")))
    }

    pub fn num_free_blocks(&self) -> i32 {
        self.alloc.num_free_blocks
    }

    /// The blocks [`CachePool::allocate_block`] may still hand out, which is what is free less the
    /// floor kept back for runs that have not been taken yet.
    pub fn num_allocatable_blocks(&self) -> i32 {
        self.alloc.num_allocatable_blocks()
    }

    pub fn num_free_runs(&self) -> i32 {
        self.alloc.free_runs.len() as i32
    }

    /// Take one block.
    pub fn allocate_block(&mut self) -> Option<i32> {
        self.alloc.allocate_block()
    }

    /// Take `count` blocks, or nothing at all when there are not that many to be had.
    ///
    /// They are not consecutive and do not need to be: what wants consecutive blocks is a run, and
    /// it asks for one. Blocks are appended to `out`, which the caller owns, so a scheduler that
    /// grows a sequence every step allocates nothing to do it.
    pub fn allocate_blocks(&mut self, count: i32, out: &mut Vec<i32>) -> bool {
        if count < 0 {
            return false;
        }
        if count > self.num_allocatable_blocks() {
            return false;
        }

        let mark = out.len();
        for _ in 0..count {
            match self.alloc.allocate_block() {
                Some(block) => out.push(block),
                None => {
                    // The count above is a bound rather than a promise -- the floor is counted in
                    // blocks and taken in runs -- so an attempt that runs out puts back what it
                    // took and reports nothing, leaving the pool as it found it.
                    for &block in &out[mark..] {
                        self.alloc.free_block(block);
                    }
                    out.truncate(mark);
                    return false;
                }
            }
        }
        true
    }

    /// Take a whole run: `blocks_per_run` consecutive blocks, named by their run number. Block
    /// `run * blocks_per_run` is the first of them.
    pub fn allocate_run(&mut self) -> Option<i32> {
        self.alloc.allocate_run()
    }

    pub fn free_block(&mut self, block: i32) -> Result<()> {
        if block < 0 || block >= self.num_blocks() {
            return Err(Error::model(format!(
                "block {block} is not one of this pool's {} blocks",
                self.num_blocks()
            )));
        }
        self.alloc.free_block(block);
        Ok(())
    }

    pub fn free_blocks(&mut self, blocks: &[i32]) -> Result<()> {
        for &block in blocks {
            self.free_block(block)?;
        }
        Ok(())
    }

    pub fn free_run(&mut self, run: i32) -> Result<()> {
        if run < 0 || run >= self.num_runs() {
            return Err(Error::model(format!(
                "run {run} is not one of this pool's {} runs",
                self.num_runs()
            )));
        }
        if self.alloc.state(run) != RunState::Full {
            return Err(Error::model(format!(
                "run {run} was not taken whole, so it cannot be given back whole"
            )));
        }
        self.alloc.free_run(run);
        Ok(())
    }

    /// Write zero over one run of every slot, which is the state a sequence that has just been
    /// handed it begins from.
    ///
    /// The run is zeroed as bytes rather than through the views of the group that took it: zero
    /// bytes are zero in every type it may be read as, and going through the storage covers the
    /// padding between the tensors as well, which nothing else ever writes.
    pub fn clear_run(&mut self, run: i32) -> Result<()> {
        if run < 0 || run >= self.num_runs() {
            return Err(Error::model(format!(
                "run {run} is not one of this pool's {} runs",
                self.num_runs()
            )));
        }

        let run_bytes = self.shape.page_size_bytes * i64::from(self.blocks_per_run);
        let num_runs = self.shape.num_pages / self.blocks_per_run;
        for layer in &mut self.storage {
            let mut bytes = layer
                .view(&[num_runs, run_bytes as i32])?
                .subtensor(run)?
                .view_as(DType::Float)?;
            F::fill(&mut bytes, 0.0)?;
        }
        Ok(())
    }
}
