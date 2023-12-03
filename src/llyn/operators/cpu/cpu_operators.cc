// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "llyn/operators/cpu/cpu_operators.h"

#include <stdlib.h>
#include <cmath>
#include <limits>
#include <memory>
#include "lymath/lymath.h"
#include "llyn/operators/cpu/apply_rotary_pos_emb.h"
#include "llyn/operators/cpu/attention.h"
#include "llyn/operators/cpu/cat.h"
#include "llyn/operators/cpu/copy.h"
#include "llyn/operators/cpu/lookup.h"
#include "llyn/operators/cpu/matmul.h"
#include "llyn/operators/cpu/mul.h"
#include "llyn/operators/cpu/print.h"
#include "llyn/operators/cpu/subtensor.h"
#include "llyn/operators/cpu/subtensor_list.h"
#include "llyn/operators/cpu/swiglu.h"
#include "llyn/operators/cpu/tensor.h"
#include "llyn/operators/cpu/cpu_tensor_data.h"
#include "llyn/operators/operators.h"
#include "llyn/tensor.h"

namespace llyn {
namespace op {
namespace cpu {

using internal::TensorData;
using internal::TensorShape;

CPUOperators::CPUOperators() {}

Tensor CPUOperators::createTensor(ly::Span<const int> shape, DType dtype) {
  return op::cpu::tensor(shape, dtype);
}

void CPUOperators::randFp32(Tensor *tensor) {
  float *data = tensor->getData<float>();
  int64_t numel = tensor->getNumEl();

  float randmax = RAND_MAX;
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = ::rand() / randmax - 0.5f;
  }
}

Tensor CPUOperators::createTensorLike(Subtensor<const float> input) {
  std::vector<Tensor::ShapeType> shape;
  for (const Shape &s : input.shape) {
    shape.push_back(s.shape);
  }

  return createTensor(shape, DType::kFloat);
}

Tensor CPUOperators::addFp32(Subtensor<const float> A, Subtensor<const float> B) {
  CHECK(isShapeMatchBroadcastB(A, B));

  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<const float> vBs = getVectorList(B);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<const float> vB = vBs.getSubtensor(j % vBs.getSize());
    Subtensor<float> vC = vCs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      vC.elem(i) = vA.elem(i) + vB.elem(i);
    }
  }

  return C;
}

Tensor CPUOperators::softmaxFp32(Subtensor<const float> A) {
  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);

  auto softmax_op = [](Subtensor<const float> A, Subtensor<float> C) {
    double sum = 0;
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      sum += std::exp(va);
    }

    double logsum = std::log(sum);
    for (int i = 0; i < A.dimension(0); ++i) {
      float va = A.elem(i);
      float &vc = C.elem(i);
      vc = static_cast<float>(std::exp(va - logsum));
    }
  };

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int i = 0; i < vAs.getSize(); ++i) {
    softmax_op(vAs.getSubtensor(i), vCs.getSubtensor(i));
  }

  return C;
}

Tensor CPUOperators::geluFp32(Subtensor<const float> A) {
  Tensor C = createTensorLike(A);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);

  auto gelu_op = [](Subtensor<const float> A, Subtensor<float> C) {
    for (int i = 0; i < A.dimension(0); ++i) {
      float x = A.elem(i);

      double x3 = pow(x, 3.0);
      double c = 0.5 * x * (1 + tanh(sqrt(2.0 / Pi) * (x + 0.044715 * x3)));
      C.elem(i) = static_cast<float>(c);
    }
  };

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int i = 0; i < vAs.getSize(); ++i) {
    gelu_op(vAs.getSubtensor(i), vCs.getSubtensor(i));
  }
  return C;
}

bool CPUOperators::allCloseFp32(
    Subtensor<const float> A, Subtensor<const float> B, float rtol, float atol) {
  CHECK(isShapeMatch(A, B));

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<const float> vBs = getVectorList(B);
  CHECK(vAs.getSize() == vBs.getSize());

  bool all_close = true;
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<const float> vB = vBs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      float va = vA.elem(i);
      float vb = vB.elem(i);
      if (!(std::isfinite(va) && std::isfinite(vb))) {
        all_close = false;
      }
      if (fabs(va - vb) > atol + rtol * fabs(vb)) {
        all_close = false;
      }
    }
  }
  
  return all_close;
}



Tensor CPUOperators::rmsNormFp32(
    Subtensor<const float> input, Subtensor<const float> weight, float eps) {
  CHECK(weight.rank() == 1);
  CHECK(input.dimension(input.rank() - 1) == weight.dimension(0));

  Tensor C = createTensorLike(input);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  SubtensorList<const float> vAs = getVectorList(input);
  SubtensorList<float> vCs = getVectorList(Cs);

  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    double sum = 0.0;
    for (int i = 0; i < vA.dimension(0); ++i) {
      double a = vA.elem(i);
      sum += a * a;
    }
    double mean = sum / vA.dimension(0);
    double rms = sqrt(mean + eps);

    // compute rms-norm
    for (int i = 0; i < vA.dimension(0); ++i) {
      float elem = static_cast<float>(vA.elem(i) / rms);
      vC.elem(i) = elem * weight.elem(i);
    }
  }

  return C;
}

Tensor CPUOperators::layerNormFp32(
    Subtensor<const float> input,
    Subtensor<const float> weight,
    Subtensor<const float> bias,
    float eps) {
  CHECK(bias.rank() == 1 && weight.rank() == 1);
  CHECK(weight.dimension(0) == bias.dimension(0));
  CHECK(input.dimension(input.rank() - 1) == weight.dimension(0));

  Tensor C = createTensorLike(input);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  SubtensorList<const float> vAs = getVectorList(input);
  SubtensorList<float> vCs = getVectorList(Cs);

  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    double sum = 0.0f;
    for (int i = 0; i < vA.dimension(0); ++i) {
      sum += vA.elem(i);
    }
    double mean = sum / vA.dimension(0);
    
    // var (unbiased)
    sum = 0.0;
    for (int i = 0; i < vA.dimension(0); ++i) {
      double d = vA.elem(i) - mean;
      sum += d * d;
    }
    double var = sum / vA.dimension(0);
    double sd = sqrt(var + eps);

    // compute layer-norm
    for (int i = 0; i < vA.dimension(0); ++i) {
      float elem = static_cast<float>((vA.elem(i) - mean) / sd); 
      vC.elem(i) = elem * weight.elem(i) + bias.elem(i);
    }
  }

  return C;
}

Tensor CPUOperators::causalMaskFp32(int seq_len) {
  Tensor mask = createTensor(ly::makeConstSpan({seq_len, seq_len}), DType::kFloat);
  CHECK(mask.isContiguous());

  float *data = mask.getData<float>();
  for (int i = 0; i < seq_len; ++i) {
    float *row = data + i * seq_len;
    for (int j = 0; j <= i; ++j) {
      row[j] = 0.0f;
    }
    for (int j = i + 1; j < seq_len; ++j) {
      row[j] = -std::numeric_limits<float>::infinity();
    }
  }

  return mask;
}

// -- class CPUOperators ----------

Tensor CPUOperators::createTensor(std::initializer_list<int> shape, DType dtype) {
  return createTensor(ly::makeConstSpan(shape), dtype);
}

Tensor CPUOperators::createTensorLike(Tensor input) {
  return createTensor(input.getShape(), input.getDType());
}

Tensor CPUOperators::rand(std::initializer_list<int> shape, DType dtype) {
  Tensor tensor = createTensor(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      randFp32(&tensor);
      break;
    default:
      CHECK(false) << "unsupported dtype for Rand";
  }

  return tensor;
}

Tensor CPUOperators::zeros(ly::Span<const int> shape, DType dtype) {
  Tensor tensor = createTensor(shape, dtype);
  switch (dtype) {
    case DType::kFloat:
      zerosFp32(Subtensor<float>::fromTensor(tensor));
      break;
    default:
      CHECK(false) << "unsupported dtype for Zeros";
  }

  return tensor;
}

Tensor CPUOperators::matmul(Tensor A, Tensor B) {
  DType typeA = A.getDType();
  DType typeB = B.getDType();
  if (typeA == DType::kFloat && typeB == DType::kFloat) {
    return matmulFp32(A, B);
  } else if (typeA == DType::kFloat && typeB == DType::kQInt4SymGroup32) {
    return matmulFp32Q4SymFp32(A, B);
  } else if (typeA == DType::kFloat && typeB == DType::kQInt4Group32) {
    return matmulFp32Q4Fp32(A, B);
  } else {
    NOT_IMPL();
    return Tensor();
  }
}

void CPUOperators::print(Tensor tensor) {
  return cpu::print(tensor);
}

Tensor CPUOperators::add(Tensor input, Tensor other) {
  switch (input.getDType()) {
    case DType::kFloat:
      return addFp32(Subtensor<const float>::fromTensor(input), Subtensor<const float>::fromTensor(other));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::softmax(Tensor input) {
  switch (input.getDType()) {
    case DType::kFloat:
      return softmaxFp32(Subtensor<const float>::fromTensor(input));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::gelu(Tensor input) {
  switch (input.getDType()) {
    case DType::kFloat:
      return geluFp32(Subtensor<const float>::fromTensor(input));
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

bool CPUOperators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  if (A.getDType() != B.getDType()) {
    return false;
  }

  switch (A.getDType()) {
    case DType::kFloat:
      return allCloseFp32(Subtensor<const float>::fromTensor(A), 
                          Subtensor<const float>::fromTensor(B),
                          rtol,
                          atol);
      break;
    default:
      NOT_IMPL();
  }

  return false;
}

Tensor CPUOperators::mul(Tensor A, float k) {
  switch (A.getDType()) {
    case DType::kFloat:
      return mulFp32(Subtensor<const float>::fromTensor(A), k);
      break;
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::mul(Tensor A, Tensor B) {
  return cpu::mul(A, B);
}

Tensor CPUOperators::lookup(Tensor table, Tensor indices) {
  return cpu::lookup(table, indices);
}

Tensor CPUOperators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  CHECK(input.getDType() == weight.getDType() && input.getDType() == bias.getDType());

  switch (input.getDType()) {
    case DType::kFloat:
      return layerNormFp32(
          Subtensor<const float>::fromTensor(input),
          Subtensor<const float>::fromTensor(weight),
          Subtensor<const float>::fromTensor(bias),
          eps);
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::rmsNorm(Tensor input, Tensor weight, float eps) {
  CHECK(input.getDType() == weight.getDType());

  switch (input.getDType()) {
    case DType::kFloat:
      return rmsNormFp32(
          Subtensor<const float>::fromTensor(input),
          Subtensor<const float>::fromTensor(weight),
          eps);
    default:
      NOT_IMPL();
  }

  return Tensor();
}

Tensor CPUOperators::causalMask(int max_len) {
  return causalMaskFp32(max_len);
}


Tensor CPUOperators::cat(Tensor A, Tensor B, int dim) {
  return cpu::cat(A, B, dim);
}

Tensor CPUOperators::applRotaryPosEmb(Tensor A, Tensor roPE) {
  return cpu::applyRotaryPosEmb(A, roPE);
}

void CPUOperators::copy(Tensor src, Tensor dest) {
  return cpu::copy(src, dest);
}

Tensor CPUOperators::attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  return cpu::attention(q, k, v, mask);
}

Tensor CPUOperators::swiglu(Tensor A) {
  return cpu::swiglu(A);
}

Tensor CPUOperators::toDevice(Tensor tensor, Device device) {
  NOT_IMPL();
}

Tensor CPUOperators::cast(Tensor tensor, DType dtype) {
  NOT_IMPL();
}

DType CPUOperators::getDefaultFloatType() {
  return DType::kFloat;
}

}  // cpu
}  // op
}  // llyn
