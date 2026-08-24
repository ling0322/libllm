/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <cuda_runtime_api.h>

#include <cutlass/numeric_types.h>

#include "flash.h"

namespace FLASH_NAMESPACE {
namespace {

template<typename T, int Headdim>
void run_mha_fwd_headdim(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    // Only the split-KV kernel resolves a block table, so paged callers force it even with 1 split.
    const bool split = params.num_splits > 1 || force_split_kernel;

    if (params.is_causal) {
        if (split) {
            run_mha_fwd_splitkv_dispatch<T, Headdim, true>(params, stream);
        } else {
            run_mha_fwd_<T, Headdim, true>(params, stream);
        }
    } else {
        if (split) {
            run_mha_fwd_splitkv_dispatch<T, Headdim, false>(params, stream);
        } else {
            run_mha_fwd_<T, Headdim, false>(params, stream);
        }
    }
}

// Only the head dimensions that have generated instantiations are listed here.
template<typename T>
bool run_mha_fwd_dtype(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    switch (params.d) {
        case 64:
            run_mha_fwd_headdim<T, 64>(params, stream, force_split_kernel);
            return true;
        case 128:
            run_mha_fwd_headdim<T, 128>(params, stream, force_split_kernel);
            return true;
        case 256:
            run_mha_fwd_headdim<T, 256>(params, stream, force_split_kernel);
            return true;
        default:
            return false;
    }
}

}  // namespace

bool run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel) {
    if (params.is_bf16) {
        return run_mha_fwd_dtype<cutlass::bfloat16_t>(params, stream, force_split_kernel);
    }

    return run_mha_fwd_dtype<cutlass::half_t>(params, stream, force_split_kernel);
}

}  // namespace FLASH_NAMESPACE
