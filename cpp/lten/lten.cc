// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include "lten.h"

#include <string.h>

#include <memory>
#include <mutex>
#include <string>

#include "lten/functional.h"
#include "lten/operators.h"
#include "lten/tensor.h"
#include "lutil/error.h"
#include "lutil/strings.h"

namespace F = lten::F;
using lten::Device;
using lten::DType;
using lten::Tensor;

namespace {

std::once_flag gLlmInitOnce;
enum LtenOpType { LBinaryOp, LBinaryScalarOp, LUnaryOp, LReduceOp };

thread_local char gErrorMessage[512] = "";

void initLTen() {
  std::call_once(gLlmInitOnce, []() {
    try {
      lut::setLogLevel(lut::LogSeverity::kINFO);
      lten::initOperators();
    } catch (const lut::Error &e) {
      LOG(ERROR) << "initialize libllm failed: " << e.what();
    }
  });
}

void llmSetErrorMessage(const std::string &message) {
  std::string what = message;
  if (what.size() >= sizeof(gErrorMessage)) {
    what.erase(what.begin() + sizeof(gErrorMessage) - 4, what.end());
    what += "...";
  }
  snprintf(gErrorMessage, sizeof(gErrorMessage), "%s", what.c_str());
}

lten::DType getDType(int32_t dtype) {
  switch (dtype) {
    case LTEN_DTYPE_FLOAT:
      return DType::kFloat;
    case LTEN_DTYPE_INT64:
      return DType::kLong;
    case LTEN_DTYPE_UINT8:
      return DType::kUInt8;
    case LTEN_DTYPE_FLOAT16:
      return DType::kFloat16;
    case LTEN_DTYPE_QINT4:
      return DType::kQInt4x32;
    case LTEN_DTYPE_INT8:
      return DType::kInt8;
    default:
      throw lut::InvalidArgError("dtype");
  }
}

int32_t fromDType(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
      return LTEN_DTYPE_FLOAT;
    case DType::kLong:
      return LTEN_DTYPE_INT64;
    case DType::kUInt8:
      return LTEN_DTYPE_UINT8;
    case DType::kFloat16:
      return LTEN_DTYPE_FLOAT16;
    case DType::kQInt4x32:
      return LTEN_DTYPE_QINT4;
    case DType::kInt8:
      return LTEN_DTYPE_INT8;
    default:
      throw lut::InvalidArgError("dtype");
  }
}

lten::Device getDevice(int32_t device) {
  if (device == LTEN_DEVICE_CPU) {
    return lten::Device::getCpu();
  } else if (device == LTEN_DEVICE_CUDA) {
    return lten::Device::getCuda();
  } else {
    throw lut::InvalidArgError("device");
  }
}

int32_t fromDevice(lten::Device device) {
  switch (device.getType()) {
    case Device::kCpu:
      return LTEN_DEVICE_CPU;
    case Device::kCuda:
      return LTEN_DEVICE_CUDA;
    default:
      throw lut::InvalidArgError("device");
  }
}

std::vector<int> getShape(int32_t dim, const int64_t *shape) {
  std::vector<int> lshape(dim);
  for (int d = 0; d < dim; ++d) {
    lshape[d] = static_cast<int>(shape[d]);
  }

  return lshape;
}

int getLtenOpTensorOperandNum(int32_t op) {
  switch (op) {
    case LTEN_OP_LAYER_NORM:
      return 3;
    case LTEN_OP_ADD:
    case LTEN_OP_MUL:
    case LTEN_OP_ROPE:
    case LTEN_OP_MATMUL:
    case LTEN_OP_RMS_NORM:
      return 2;
    case LTEN_OP_SUM:
    case LTEN_OP_MAX:
    case LTEN_OP_SOFTMAX:
    case LTEN_OP_GELU:
    case LTEN_OP_SWIGLU:
    case LTEN_OP_CONTIGUOUS:
    case LTEN_OP_SCALAR_MUL:
      return 1;
    default:
      throw lut::InvalidArgError(lut::sprintf("unsupported binary operator: %d", op));
  }
}

}  // namespace

struct LTensor {
  lten::Tensor tensorl;
};

const char *lten_last_error_message() {
  return gErrorMessage;
}

int32_t lten_destroy_tensor(LTensor *tensor) {
  try {
    delete tensor;
    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

LTensor *lten_new_tensor(int32_t dim, const int64_t *shape, int32_t dtype, int32_t device) {
  initLTen();

  try {
    if (!shape) throw lut::InvalidArgError("shape");
    lten::Device devicel = getDevice(device);
    lten::DType dtypel = getDType(dtype);

    std::unique_ptr<LTensor> tensor = std::make_unique<LTensor>();
    std::vector<int> shapel = getShape(dim, shape);
    tensor->tensorl = F::tensor(shapel, dtypel, devicel);

    return tensor.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

int32_t lten_get_dim(LTensor *tensor, int32_t *dim) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    *dim = tensor->tensorl.getDim();

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

int32_t lten_get_shape(LTensor *tensor, int32_t dim, int64_t *size) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    *size = tensor->tensorl.getShape(dim);

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

int32_t lten_get_dtype(LTensor *tensor, int32_t *dtype) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    DType dtypel = tensor->tensorl.getDType();
    *dtype = fromDType(dtypel);

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

int32_t lten_get_device(LTensor *tensor, int32_t *device) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    Device d = tensor->tensorl.getDevice();
    *device = fromDevice(d);

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

LTensor *lten_view(LTensor *tensor, int32_t dim, int64_t *shape) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    if (!shape) throw lut::InvalidArgError("shape");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    std::vector<int> shapel = getShape(dim, shape);
    out->tensorl = tensor->tensorl.view(shapel);

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_transpose(LTensor *tensor, int32_t dim0, int32_t dim1) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = tensor->tensorl.transpose(dim0, dim1);

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_expand(LTensor *tensor, int32_t dim, int64_t *shape) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    if (!shape) throw lut::InvalidArgError("shape");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    std::vector<int> shapel = getShape(dim, shape);
    out->tensorl = tensor->tensorl.expand(shapel);

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_slice(LTensor *tensor, int32_t dim, int64_t begin, int64_t end) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    if (begin == LTEN_RANGE_NONE) begin = 0;
    if (end == LTEN_RANGE_NONE) end = lten::None;

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = tensor->tensorl.slice(
        dim,
        std::make_pair<int, int>(static_cast<int>(begin), static_cast<int>(end)));

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_index(LTensor *tensor, int64_t index) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = tensor->tensorl.subtensor(index);

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_to_device(LTensor *tensor, int32_t device) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = F::to(getDevice(device), tensor->tensorl);

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_to_dtype(LTensor *tensor, int32_t dtype) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = F::cast(tensor->tensorl, getDType(dtype));

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

int32_t lten_copy(LTensor *dest, LTensor *src) {
  try {
    if (!dest) throw lut::InvalidArgError("dest");
    if (!src) throw lut::InvalidArgError("src");
    F::copy(src->tensorl, dest->tensorl);

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

int32_t lten_get_numel(LTensor *tensor, int64_t *numel) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    if (!numel) throw lut::InvalidArgError("numel");
    *numel = tensor->tensorl.getNumEl();

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

void *lten_get_data_ptr(LTensor *tensor) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    Tensor x = tensor->tensorl;
    if (x.getDevice().getType() != Device::kCpu) {
      throw lut::AbortedError("get data ptr supports CPU tensor");
    }
    if (!x.isContiguous()) {
      throw lut::AbortedError("get data ptr only supports contiguous tensor");
    }

    return x.getData<void>();

  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

int32_t lten_fill_float(LTensor *tensor, float value) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    Tensor x = tensor->tensorl;
    F::fill(x, value);

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

int32_t lten_print(LTensor *tensor) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");
    Tensor x = tensor->tensorl;
    F::print(x);

    return 0;
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return static_cast<int32_t>(e.getCode());
  }
}

LTensor *lten_clone(LTensor *tensor) {
  try {
    if (!tensor) throw lut::InvalidArgError("tensor");

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = tensor->tensorl;

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}

LTensor *lten_apply_operator(
    LTensor *targ0,
    LTensor *targ1,
    LTensor *targ2,
    LTensor *targ3,
    int64_t iarg0,
    int64_t iarg1,
    float farg0,
    float farg1,
    int32_t op) {
  try {
    int numOperands = getLtenOpTensorOperandNum(op);
    if (numOperands >= 1 && !targ0) throw lut::InvalidArgError("targ0");
    if (numOperands >= 2 && !targ1) throw lut::InvalidArgError("targ1");
    if (numOperands >= 3 && !targ2) throw lut::InvalidArgError("targ2");
    if (numOperands >= 4 && !targ3) throw lut::InvalidArgError("targ3");

    Tensor c;
    int iiarg0 = static_cast<int>(iarg0);
    switch (op) {
      case LTEN_OP_ADD:
        c = F::add(targ0->tensorl, targ1->tensorl);
        break;
      case LTEN_OP_MUL:
        c = F::mul(targ0->tensorl, targ1->tensorl);
        break;
      case LTEN_OP_ROPE:
        c = F::applyRotaryPosEmb(targ0->tensorl, targ1->tensorl);
        break;
      case LTEN_OP_SOFTMAX:
        c = F::softmax(targ0->tensorl);
        break;
      case LTEN_OP_GELU:
        c = F::gelu(targ0->tensorl);
        break;
      case LTEN_OP_SWIGLU:
        c = F::swiglu(targ0->tensorl);
        break;
      case LTEN_OP_CONTIGUOUS:
        c = F::contiguous(targ0->tensorl);
        break;
      case LTEN_OP_SUM:
        c = F::sum(targ0->tensorl, iiarg0);
        break;
      case LTEN_OP_MAX:
        c = F::max(targ0->tensorl, iiarg0);
        break;
      case LTEN_OP_MATMUL:
        c = F::matmul(targ0->tensorl, targ1->tensorl);
        break;
      case LTEN_OP_LOOKUP:
        c = F::lookup(targ0->tensorl, targ1->tensorl);
        break;
      case LTEN_OP_SCALAR_MUL:
        c = F::mul(targ0->tensorl, farg0);
        break;
      case LTEN_OP_LAYER_NORM:
        c = F::layerNorm(targ0->tensorl, targ1->tensorl, targ2->tensorl, farg0);
        break;
      case LTEN_OP_RMS_NORM:
        c = F::rmsNorm(targ0->tensorl, targ1->tensorl, farg0);
        break;
      default:
        throw lut::InvalidArgError(lut::sprintf("unsupported binary operator: %d", op));
    }

    std::unique_ptr<LTensor> out = std::make_unique<LTensor>();
    out->tensorl = c;

    return out.release();
  } catch (const lut::Error &e) {
    llmSetErrorMessage(e.what());
    return nullptr;
  }
}
