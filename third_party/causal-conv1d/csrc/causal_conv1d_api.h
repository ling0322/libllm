/******************************************************************************
 * Not upstream. The surface flint/cuda calls; keeps ConvParamsBase out of the
 * engine's headers and the CUDA instantiations behind one translation unit.
 ******************************************************************************/

#pragma once

#include <cuda_runtime_api.h>

#include "causal_conv1d.h"

namespace causal_conv1d {

/// Forward pass over a channel-last activation, (batch, seqlen, dim). Set
/// params.seq_idx_ptr to label each position with its sequence and the kernel keeps the
/// convolution from reading across a boundary.
bool channellast_fwd(ConvParamsBase &params, bool is_half, cudaStream_t stream);

/// One decoding step, advancing params.conv_state_ptr in place.
bool update(ConvParamsBase &params, bool is_half, cudaStream_t stream);

}  // namespace causal_conv1d
