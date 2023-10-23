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

#include "llyn/util.h"

#include "lyutil/error.h"

namespace llyn {

void readParameters(const std::string &model_path, nn::Module *module) {
  StateMap state_dict;
  state_dict.read(model_path);

  module->initParameters(state_dict);
}

std::vector<Tensor> readAllTensors(const std::string &filename) {
  std::vector<Tensor> tensors;

  std::unique_ptr<ly::ReadableFile> fp = ly::ReadableFile::open(filename);
  for (; ; ) {
    Tensor A;
    try {
      A.read(fp.get());
    } catch (const ly::OutOfRangeError &) {
      break;
    }
    
    tensors.emplace_back(A);
  }

  return tensors;
}

Context getCtxForCPU() {
  Context ctx;
  ctx.setDevice(Device::createForCPU());

  return ctx;
}

}