#pragma once

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

Tensor sample(
	const Tensor &distribution,
	const Tensor &uniformNoise,
	int topK,
	float topP);

}  // namespace cuda
}  // namespace op
}  // namespace fl