/******************************************************************************
 * Not upstream. Replaces causal_conv1d.cpp, which bound the kernels to PyTorch:
 * it took at::Tensor, read its sizes and strides, and dispatched on its dtype.
 * The engine has its own tensor type, so it fills in ConvParamsBase itself and
 * this file is only the dtype dispatch that the .cpp also carried.
 ******************************************************************************/

#include "causal_conv1d.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

template<typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

namespace causal_conv1d {

bool channellast_fwd(ConvParamsBase &params, bool is_half, cudaStream_t stream) {
  // Only the instantiations the vendored .cu files carry: input and weight of the same type,
  // either half or float. A mixed pair would link against a symbol that is not generated.
  if (is_half) {
    causal_conv1d_channellast_fwd_cuda<__half, __half>(params, stream);
  } else {
    causal_conv1d_channellast_fwd_cuda<float, float>(params, stream);
  }

  return true;
}

bool update(ConvParamsBase &params, bool is_half, cudaStream_t stream) {
  if (is_half) {
    causal_conv1d_update_cuda<__half, __half>(params, stream);
  } else {
    causal_conv1d_update_cuda<float, float>(params, stream);
  }

  return true;
}

}  // namespace causal_conv1d
