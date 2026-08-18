/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <cuda_runtime_api.h>

#include <cutlass/numeric_types.h>

#include "flash.h"

namespace FLASH_NAMESPACE {
namespace {

template<typename T, int Headdim>
void run_mha_fwd_headdim(Flash_fwd_params &params, cudaStream_t stream) {
    const bool split = params.num_splits > 1;

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
bool run_mha_fwd_dtype(Flash_fwd_params &params, cudaStream_t stream) {
    switch (params.d) {
        case 64:
            run_mha_fwd_headdim<T, 64>(params, stream);
            return true;
        case 128:
            run_mha_fwd_headdim<T, 128>(params, stream);
            return true;
        case 256:
            run_mha_fwd_headdim<T, 256>(params, stream);
            return true;
        default:
            return false;
    }
}

}  // namespace

bool run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_bf16) {
        return run_mha_fwd_dtype<cutlass::bfloat16_t>(params, stream);
    }

    return run_mha_fwd_dtype<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
