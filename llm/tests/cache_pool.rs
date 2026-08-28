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

//! The pool that hands out blocks, and runs of blocks, of every layer at once.

use llm::flint::{functional as F, DType, Device};
use llm::{BlockShape, CachePool, CacheUnit, GroupShape, MAX_BLOCKS_PER_RUN};

const LAYERS: i32 = 3;
const BLOCKS_PER_RUN: i32 = 4;
const RUNS: i32 = 4;

/// What a page has to be for [`blocks`]: (2*2 + 3) floats, rounded up to the alignment.
const PAGE: i64 = 32;

/// A group that takes a block at a time and keeps two tensors in it, the way an attention layer
/// keeps the keys and values of a fixed number of tokens.
fn blocks() -> GroupShape {
    GroupShape::new(vec![vec![2, 2], vec![3]], DType::Float, CacheUnit::Block).unwrap()
}

/// A group that takes a whole run and keeps two tensors across it, the way a gated DeltaNet layer
/// keeps a recurrence and a convolution window. (4*6 + 8) floats is four pages.
fn runs() -> GroupShape {
    GroupShape::new(vec![vec![4, 6], vec![8]], DType::Float, CacheUnit::Run).unwrap()
}

/// A pool of `runs` runs of the two groups, whose page and run length come out of their shapes.
fn pool_of(groups: Vec<GroupShape>, num_runs: i32, reserved: i32) -> CachePool {
    let (page_size_bytes, blocks_per_run) = CachePool::page_layout(&groups).unwrap();
    let shape = BlockShape::new(page_size_bytes, LAYERS, num_runs * blocks_per_run).unwrap();
    CachePool::new(shape, groups, reserved, Device::Cpu).unwrap()
}

fn pool(num_runs: i32, reserved: i32) -> CachePool {
    pool_of(vec![blocks(), runs()], num_runs, reserved)
}

#[test]
fn a_block_names_the_same_page_of_every_layer() {
    let pool = pool(RUNS, 0);

    assert_eq!(pool.num_layers(), LAYERS);
    assert_eq!(pool.num_groups(), 2);
    assert_eq!(pool.num_blocks(), RUNS * BLOCKS_PER_RUN);
    assert_eq!(pool.num_runs(), RUNS);
    assert_eq!(pool.blocks_per_run(), BLOCKS_PER_RUN);

    // A page is what the group that takes blocks needs of one, and the run is as many pages as the
    // group that takes runs needs.
    let shape = pool.shape();
    assert_eq!(shape.page_size_bytes, PAGE);
    assert_eq!(shape.num_layers, LAYERS);
    assert_eq!(shape.num_pages, RUNS * BLOCKS_PER_RUN);

    // A block costs a page in every layer, and the pool is that many blocks.
    assert_eq!(pool.bytes_per_block(), PAGE * i64::from(LAYERS));
    assert_eq!(
        pool.bytes(),
        PAGE * i64::from(LAYERS * RUNS * BLOCKS_PER_RUN)
    );

    // Every layer is as long as every other, which is what lets one id name a block in all of
    // them, and reads its own tensors at its own shapes: the group that takes blocks is indexed by
    // the block, the one that takes runs by the run.
    for slot in 0..LAYERS {
        let tensors = pool.layer(0, slot).unwrap();
        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].shape(), vec![RUNS * BLOCKS_PER_RUN, 2, 2]);
        assert_eq!(tensors[1].shape(), vec![RUNS * BLOCKS_PER_RUN, 3]);

        let tensors = pool.layer(1, slot).unwrap();
        assert_eq!(tensors[0].shape(), vec![RUNS, 4, 6]);
        assert_eq!(tensors[1].shape(), vec![RUNS, 8]);
    }
    assert!(pool.layer(0, LAYERS).is_err(), "no such slot");
    assert!(pool.layer(2, 0).is_err(), "no such group");
}

#[test]
fn the_pool_holds_whole_runs_only() {
    // Two pages more than three runs is three runs: the pages at the end could never be handed to
    // the group that takes runs, and the allocator's bookkeeping is one bitmap per run.
    let shape = BlockShape::new(PAGE, LAYERS, 3 * BLOCKS_PER_RUN + 2).unwrap();
    let pool = CachePool::new(shape, vec![blocks(), runs()], 0, Device::Cpu).unwrap();

    assert_eq!(pool.num_runs(), 3);
    assert_eq!(pool.num_blocks(), 3 * BLOCKS_PER_RUN);
    assert_eq!(pool.shape().num_pages, 3 * BLOCKS_PER_RUN);
}

#[test]
fn a_larger_page_makes_a_shorter_run() {
    // The page size is the caller's to choose: page_layout says the smallest one that serves the
    // groups, and one twice that fits a state in half the blocks.
    let groups = vec![blocks(), runs()];
    assert_eq!(CachePool::page_layout(&groups).unwrap(), (PAGE, 4));
    assert_eq!(CachePool::run_length(&groups, 2 * PAGE).unwrap(), 2);

    let shape = BlockShape::new(2 * PAGE, LAYERS, 8).unwrap();
    let pool = CachePool::new(shape, groups, 0, Device::Cpu).unwrap();
    assert_eq!(pool.blocks_per_run(), 2);
    assert_eq!(pool.num_runs(), 4);

    // A pool of nothing but runs has no page size of its own, so a state is one page.
    assert_eq!(CachePool::page_layout(&[runs()]).unwrap(), (128, 1));
}

#[test]
fn blocks_come_out_of_one_run_until_it_is_full() {
    let mut pool = pool(RUNS, 0);

    // Reading the partial stack from the top keeps the traffic inside one run, so the runs that
    // have not been broken into stay whole and a large allocation can still be served.
    let first: Vec<i32> = (0..BLOCKS_PER_RUN)
        .map(|_| pool.allocate_block().unwrap())
        .collect();
    assert_eq!(first, vec![0, 1, 2, 3]);
    assert_eq!(pool.num_free_runs(), RUNS - 1);

    // Only once it is full does the next block break a second run open.
    assert_eq!(pool.allocate_block(), Some(4));
    assert_eq!(pool.num_free_runs(), RUNS - 2);
}

#[test]
fn a_run_is_consecutive_blocks_named_by_its_number() {
    let mut pool = pool(RUNS, 0);

    let run = pool.allocate_run().unwrap();
    assert_eq!(run, 0);
    assert_eq!(pool.num_free_blocks(), (RUNS - 1) * BLOCKS_PER_RUN);

    // Blocks run * blocks_per_run .. + blocks_per_run, consecutively, which is what lets the same
    // bytes be read as one large allocation by its run number without an indirection.
    let second = pool.allocate_run().unwrap();
    assert_eq!(second, 1);

    // What a run took is not available a block at a time.
    let block = pool.allocate_block().unwrap();
    assert!(
        block >= 2 * BLOCKS_PER_RUN,
        "block {block} came out of a taken run"
    );
}

#[test]
fn a_run_emptied_a_block_at_a_time_is_whole_again() {
    let mut pool = pool(RUNS, 0);

    // Break one run into blocks, then give every one of them back. The run has to find its way
    // back to the free stack, or a pool that has served enough short sequences would have no whole
    // run left for a long one even with everything free.
    let blocks: Vec<i32> = (0..BLOCKS_PER_RUN)
        .map(|_| pool.allocate_block().unwrap())
        .collect();
    assert_eq!(pool.num_free_runs(), RUNS - 1);

    for &block in &blocks {
        pool.free_block(block).unwrap();
    }
    assert_eq!(pool.num_free_runs(), RUNS);
    assert_eq!(pool.num_free_blocks(), RUNS * BLOCKS_PER_RUN);

    // And it can be handed out whole, which is the case the stale entry on the partial stack has
    // to not get in the way of.
    for expected in 0..RUNS {
        assert_eq!(pool.allocate_run(), Some(expected));
    }
    assert_eq!(pool.allocate_run(), None);
    assert_eq!(pool.allocate_block(), None);
}

#[test]
fn a_half_emptied_run_is_neither_whole_nor_gone() {
    let mut pool = pool(RUNS, 0);

    let blocks: Vec<i32> = (0..BLOCKS_PER_RUN)
        .map(|_| pool.allocate_block().unwrap())
        .collect();
    pool.free_block(blocks[1]).unwrap();
    pool.free_block(blocks[2]).unwrap();

    // Two of its blocks are free, so it is not a run anybody can take whole...
    assert_eq!(pool.num_free_runs(), RUNS - 1);
    // ...but those two are the next blocks handed out, lowest bit first.
    assert_eq!(pool.allocate_block(), Some(blocks[1]));
    assert_eq!(pool.allocate_block(), Some(blocks[2]));
}

#[test]
fn the_floor_keeps_whole_runs_for_the_size_that_needs_them() {
    // Two of the four runs are kept back: a request that cannot get another block can be preempted
    // and resumed, where one that cannot get its state cannot run at all.
    let mut pool = pool(RUNS, 2);

    assert_eq!(pool.num_free_blocks(), RUNS * BLOCKS_PER_RUN);
    assert_eq!(pool.num_allocatable_blocks(), 2 * BLOCKS_PER_RUN);

    let mut blocks = Vec::new();
    assert!(pool.allocate_blocks(2 * BLOCKS_PER_RUN, &mut blocks));
    assert_eq!(blocks.len(), (2 * BLOCKS_PER_RUN) as usize);

    // The floor is where the blocks stop, even though half the pool is free.
    assert_eq!(pool.allocate_block(), None);
    assert_eq!(pool.num_free_blocks(), 2 * BLOCKS_PER_RUN);

    // And it is exactly what the runs get.
    assert!(pool.allocate_run().is_some());
    assert!(pool.allocate_run().is_some());
    assert_eq!(pool.allocate_run(), None);
}

#[test]
fn a_batch_that_does_not_fit_leaves_the_pool_alone() {
    let mut pool = pool(RUNS, 0);

    let mut blocks = vec![-1; 2];
    assert!(!pool.allocate_blocks(RUNS * BLOCKS_PER_RUN + 1, &mut blocks));

    // Nothing taken, and what the caller already had in the buffer is untouched.
    assert_eq!(blocks, vec![-1, -1]);
    assert_eq!(pool.num_free_blocks(), RUNS * BLOCKS_PER_RUN);
    assert_eq!(pool.num_free_runs(), RUNS);

    // A batch that fits appends to the same buffer.
    assert!(pool.allocate_blocks(3, &mut blocks));
    assert_eq!(blocks, vec![-1, -1, 0, 1, 2]);
}

#[test]
fn a_run_of_one_is_a_plain_stack_of_blocks() {
    // Which is what a pool whose layers all take blocks asks for: every run is a block, the
    // partial stack is never used, and the lowest id comes out first.
    let mut pool = pool_of(vec![blocks()], 8, 0);

    assert_eq!(pool.blocks_per_run(), 1);
    assert_eq!(pool.num_blocks(), 8);
    assert_eq!(pool.num_runs(), 8);

    let mut blocks = Vec::new();
    assert!(pool.allocate_blocks(3, &mut blocks));
    assert_eq!(blocks, vec![0, 1, 2]);

    pool.free_blocks(&blocks).unwrap();
    assert_eq!(pool.num_free_blocks(), 8);
}

#[test]
fn a_run_is_given_back_whole_or_not_at_all() {
    let mut pool = pool(RUNS, 0);

    let run = pool.allocate_run().unwrap();
    pool.free_run(run).unwrap();
    assert_eq!(pool.num_free_runs(), RUNS);

    // Twice would hand the same bytes to two holders.
    assert!(pool.free_run(run).is_err());

    // And a run that was broken into blocks was never taken whole, so it cannot be returned whole.
    pool.allocate_block().unwrap();
    assert!(pool.free_run(0).is_err());

    assert!(pool.free_run(-1).is_err());
    assert!(pool.free_run(RUNS).is_err());
    assert!(pool.free_block(RUNS * BLOCKS_PER_RUN).is_err());
    assert!(pool.free_block(-1).is_err());
}

#[test]
fn clearing_a_run_clears_every_page_of_it() {
    let mut pool = pool(RUNS, 0);

    // Write through the group that takes blocks, which is the same bytes the group that takes runs
    // reads as its state -- a block belongs to one group at a time, so this is what the run has to
    // be rid of before it is handed to the other.
    let run = pool.allocate_run().unwrap();
    for slot in 0..LAYERS {
        for tensor in pool.layer_mut(0, slot).unwrap() {
            for block in 0..BLOCKS_PER_RUN {
                let mut view = tensor.subtensor(run * BLOCKS_PER_RUN + block).unwrap();
                F::fill(&mut view, 5.0).unwrap();
            }
        }
    }

    pool.clear_run(run).unwrap();

    for slot in 0..LAYERS {
        for tensor in pool.layer(1, slot).unwrap() {
            let values = tensor.subtensor(run).unwrap().to_vec_f32().unwrap();
            assert!(values.iter().all(|&x| x == 0.0), "slot {slot} is not zero");
        }
        for tensor in pool.layer(0, slot).unwrap() {
            for block in 0..BLOCKS_PER_RUN {
                let values = tensor
                    .subtensor(run * BLOCKS_PER_RUN + block)
                    .unwrap()
                    .to_vec_f32()
                    .unwrap();
                assert!(values.iter().all(|&x| x == 0.0), "slot {slot} is not zero");
            }
        }
    }

    // The rest of the pool is not touched by clearing one run of it.
    let other = pool.allocate_run().unwrap();
    assert_ne!(other, run);
}

#[test]
fn a_pool_has_to_be_describable() {
    let shape = BlockShape::new(PAGE, LAYERS, RUNS * BLOCKS_PER_RUN).unwrap();
    let groups = vec![blocks(), runs()];

    assert!(
        CachePool::new(shape, Vec::new(), 0, Device::Cpu).is_err(),
        "no groups"
    );
    assert!(BlockShape::new(0, LAYERS, 4).is_err(), "no bytes in a page");
    assert!(BlockShape::new(PAGE, 0, 4).is_err(), "no layers");
    assert!(BlockShape::new(PAGE, LAYERS, 0).is_err(), "no pages");
    assert!(
        BlockShape::new(1 << 20, LAYERS, 1 << 12).is_err(),
        "a layer larger than a tensor dimension"
    );

    // A page too small for what a group keeps in one, and a pool too short for one whole run.
    assert!(CachePool::run_length(&groups, PAGE / 2).is_err());
    assert!(CachePool::new(
        BlockShape::new(PAGE, LAYERS, BLOCKS_PER_RUN - 1).unwrap(),
        groups.clone(),
        0,
        Device::Cpu
    )
    .is_err());
    assert!(
        CachePool::new(shape, groups, RUNS + 1, Device::Cpu).is_err(),
        "a floor higher than the pool"
    );

    // The longest run the bitmap holds is allowed, and one page more is not.
    let widest = |pages: i32| {
        GroupShape::new(
            vec![vec![pages * PAGE as i32 / 4]],
            DType::Float,
            CacheUnit::Run,
        )
        .unwrap()
    };
    assert_eq!(
        CachePool::run_length(&[blocks(), widest(MAX_BLOCKS_PER_RUN)], PAGE).unwrap(),
        MAX_BLOCKS_PER_RUN
    );
    assert!(CachePool::run_length(&[blocks(), widest(MAX_BLOCKS_PER_RUN + 1)], PAGE).is_err());

    assert!(GroupShape::new(Vec::new(), DType::Float, CacheUnit::Block).is_err());
    assert!(GroupShape::new(vec![vec![2, 0]], DType::Float, CacheUnit::Block).is_err());
}
