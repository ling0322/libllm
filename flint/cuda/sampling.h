#pragma once

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

Tensor sample(
	const Tensor &logits,
	const Tensor &uniformNoise,
	const Tensor &temperatures,
	const Tensor &topKs,
	const Tensor &topPs);

}  // namespace cuda
}  // namespace op
}  // namespace fl