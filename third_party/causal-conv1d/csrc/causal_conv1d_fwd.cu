/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "causal_conv1d_compat.h"

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #define ROCM_ONLY(x) 
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
    #define ROCM_ONLY(x) x
#endif

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"
#include "static_switch.h"


template<int kNThreads_, int kWidth_, int kChunkSizeL_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_channellast_fwd_kernel_traits {
    // The cache line is 128 bytes, and we try to read 16 bytes per thread.
    // So we have 8 threads per "row", so 32 or 64 elements in the channel dimension.
    // That leaves 4 columns per warp, and so 16 columns per block (assuming each block has 128
    // threads). Each each load is 16 x 32|64 elements in the L x C dimensions.
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 32 == 0);
    static constexpr int kNWarps = kNThreads / 32;
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static constexpr int kNEltsPerRow = 128 / kNBytes;
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;  // Always 8 for now
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128);
    static constexpr int kNColsPerWarp = 32 / kNThreadsPerRow;  // Always 4 for now
    static_assert(kNColsPerWarp * kNThreadsPerRow == 32);
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    // using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // static constexpr int kSmemSize = std::max({sizeof(typename BlockLoadT::TempStorage),
    //                                            sizeof(typename BlockStoreT::TempStorage)});
    // static constexpr int kSmemSize = kChunkSizeL * kNEltsPerRow * kNBytes;
};

template<typename Ktraits, bool kHasSeqIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_channellast_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNWarp = Ktraits::kNWarps;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL][kChunkSizeC + kNElts];

    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.out_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    int *seq_idx = !kHasSeqIdx ? nullptr : reinterpret_cast<int *>(params.seq_idx_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;
    input_t *initial_states = params.initial_states_ptr == nullptr || chunk_l_id > 0 ? nullptr
        : reinterpret_cast<input_t *>(params.initial_states_ptr) + batch_id * params.initial_states_batch_stride + l_idx * params.initial_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    // The last L-chunk will also have enough info to write to final states, since it also contain a few x values
    // from the previous L-chunk.
    input_t *final_states = params.final_states_ptr == nullptr || chunk_l_id < gridDim.y - 1 ? nullptr
        : reinterpret_cast<input_t *>(params.final_states_ptr) + batch_id * params.final_states_batch_stride + l_idx * params.final_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    // Load the elements from the previous chunk that are needed for convolution.
    if (l_idx < kWidth - 1) {
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
            && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x - (kWidth - 1) * params.x_l_stride);
        } else if (initial_states != nullptr
                   && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < 0
                   && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(initial_states);
        }
        reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    __syncthreads();

    if (final_states != nullptr
        && l_idx < kWidth - 1
        && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
        // x_smem[0] contains element at index chunk_l_id * kChunkSizeL - (kWidth - 1)
        // So last few elements (index params.seqlen - kWidth + 1 + l_idx) are stored in x_smem[params.seqlen - kWidth + 1 + l_idx - (chunk_l_id * kChunkSizeL - kWidth + 1)][c_idx]
        *reinterpret_cast<vec_t *>(final_states) = reinterpret_cast<vec_t *>(x_smem[params.seqlen + l_idx - chunk_l_id * kChunkSizeL])[c_idx];
    }

    constexpr int kLPerThread = constexpr_min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    // kChunkSizeL, kLPerThread, kNThreadsPerRow should be powers of 2 for simplicity
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 32);

    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;

    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = weight[row_idx * params.weight_c_stride + w * params.weight_width_stride];
        }
    }
    float x_vals[kWidth - 1 + kLPerThread];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }
    int seq_idx_thread[kWidth - 1 + kLPerThread];
    if constexpr (kHasSeqIdx) {
        const int initial_state_seq_idx = initial_states == nullptr ? -1 : seq_idx[0];
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            const int l_idx = chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i - (kWidth - 1);
            seq_idx_thread[i] = l_idx >= 0 && l_idx < params.seqlen
                ? seq_idx[col_idx * kLPerThread + i - (kWidth - 1)]
                : l_idx < 0 ? initial_state_seq_idx : -1;
        }
    }

    float out_vals[kLPerThread];
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        out_vals[i] = bias_val;
        const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
        // For padding tokens (seq_idx < 0), we skip the computation and set the output to 0.
        if (seq_idx_cur < 0) {
            out_vals[i] = 0.f;
            continue;
        }
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            if constexpr (!kHasSeqIdx) {
                out_vals[i] += weight_vals[w] * x_vals[i + w];
            } else {
                out_vals[i] += seq_idx_thread[i + w] == seq_idx_cur ? weight_vals[w] * x_vals[i + w] : 0.f;
            }
        }
        if (params.silu_activation) {out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i])); }
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) { x_smem[col_idx * kLPerThread + i][row_idx] = out_vals[i]; }
    __syncthreads();

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t out_vals_store[kNElts];
        reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[l * kLPerLoad + l_idx])[c_idx];
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(out + l * kLPerLoad * params.out_l_stride) = reinterpret_cast<vec_t *>(out_vals_store)[0];
        }
    }

}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seq_idx_ptr != nullptr, kHasSeqIdx, [&] {
        using Ktraits = Causal_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, 64, true, input_t, weight_t>;
        // constexpr int kSmemSize = Ktraits::kSmemSize;
        constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
        constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
        const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
        const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
        dim3 grid(params.batch, n_chunks_L, n_chunks_C);
        dim3 block(Ktraits::kNThreads);
        auto kernel = &causal_conv1d_channellast_fwd_kernel<Ktraits, kHasSeqIdx>;
        // if (kSmemSize >= 48 * 1024) {
        //     C10_CUDA_CHECK(cudaFuncSetAttribute(
        //         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        //     }
        // kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_channellast_fwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_channellast_fwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_channellast_fwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}
template void causal_conv1d_channellast_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<__half, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<float, __half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<__half, __half>(ConvParamsBase &params, cudaStream_t stream);
