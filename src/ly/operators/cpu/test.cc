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

#include "../../../../third_party/catch2/catch_amalgamated.hpp"

#include "ly/functional.h"
#include "ly/operators/cpu/fingerprint.h"
#include "ly/ly.h"

 
namespace F = ly::functional;

namespace ly {
namespace op {
namespace cpu {

Tensor RefMatMulFp32(const Tensor &A, const Tensor &B) {
  CATCH_REQUIRE(A.getDType() == B.getDType());
  CATCH_REQUIRE(A.getDim() == 2);
  CATCH_REQUIRE(B.getDim() == 2);
  CATCH_REQUIRE(A.getShape(1) == B.getShape(0));
  CATCH_REQUIRE(A.getDType() == DType::kFloat);

  Tensor C = F::zeros({A.getShape(0), B.getShape(1)}, DType::kFloat);
  float *dataC = C.getData<float>();
  const float *dataA = A.getData<float>(),
              *dataB = B.getData<float>();
  int stride0A = A.getStride(0);
  int stride1A = A.getStride(1);
  int stride0B = B.getStride(0);
  int stride1B = B.getStride(1);
  int ldc = C.getStride(0);

  for (int m = 0; m < A.getShape(0); ++m) {
    for (int n = 0; n < B.getShape(1); ++n) {
      for (int k = 0; k < A.getShape(1); ++k) {
        float va = dataA[stride0A * m + k * stride1A];
        float vb = dataB[stride0B * k + n * stride1B];
        dataC[ldc * m + n] += va * vb;
      }
    }
  }

  return C;
}

void testGEMM(int m, int k, int n, bool transa, bool transb) {
  Tensor A = transa ? F::rand({k, m}, DType::kFloat)
                    : F::rand({m, k}, DType::kFloat);
  Tensor B = transb ? F::rand({n, k}, DType::kFloat)
                    : F::rand({k, n}, DType::kFloat);

  if (transa) A = A.transpose(0, 1);
  if (transb) B = B.transpose(0, 1);

  Tensor C = F::matmul(A, B);
  Tensor C_ref = RefMatMulFp32(A, B);

  CATCH_REQUIRE(F::allClose(C, C_ref));
}

void testGEMV(int M, int N, bool TransA) {
  Tensor A = TransA ? F::rand({N, M}, DType::kFloat) : F::rand({M, N}, DType::kFloat);
  Tensor x = F::rand({N, 1}, DType::kFloat);

  if (TransA) A = A.transpose(0, 1);

  Tensor C = F::matmul(A, x);
  Tensor C_ref = RefMatMulFp32(A, x);

  CATCH_REQUIRE(F::allClose(C, C_ref));
}

int gemmTestShapes[][3] = {
  {50, 50, 1},
  {1, 1, 1},
  {2, 2, 2},
  {50, 50, 1},
  {513, 2, 513},
  {200, 1, 300},
  {1, 200, 300},
  {200, 300, 1},
  {16, 16, 5000},
  {16, 5000, 16},
  {16, 512, 16},
  {16, 1024, 16},
  {5000, 16, 16},
  {0, 0, 0}
};

CATCH_TEST_CASE("float32 GEMM BVT", "[core][nn][gemm]") {
  int (*pshape)[3];
  
  for (pshape = &gemmTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    testGEMM(m, k, n, false, false);
    testGEMM(m, k, n, true, false);
    testGEMM(m, k, n, false, true);
  }
}

int gemvTestShapes[][2] = {
  {2, 8},
  {50, 10},
  {1, 1},
  {1024, 3}
};

CATCH_TEST_CASE("float32 GEMV BVT", "[core][nn][gemv]") {
  int (*pshape)[2];
  
  for (pshape = &gemvTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int n = (*pshape)[1];

    testGEMV(m, n, false);
    testGEMV(m, n, true);
  }
}

CATCH_TEST_CASE("test embedding lookup", "[core][nn][operators]") {
  Context ctx = Context::getCpu();

  Tensor wte = Tensor::create<float>({5, 2}, {
      0.1f, 0.2f,
      0.3f, 0.4f,
      0.2f, 0.3f,
      0.4f, 0.5f,
      0.7f, 0.8f,
  });
  Tensor input = Tensor::create<LongType>({2, 3}, {
      0, 1, 2,
      1, 3, 4,
  });
  Tensor output = Tensor::create<float>({2, 3, 2}, {
      0.1f, 0.2f,
      0.3f, 0.4f,
      0.2f, 0.3f,

      0.3f, 0.4f,
      0.4f, 0.5f,
      0.7f, 0.8f,
  });
  CATCH_REQUIRE(F::allClose(F::lookup(wte, input), output));
}

CATCH_TEST_CASE("test softmax", "[core][nn][operators]") {
  Context ctx = Context::getCpu();

  Tensor input = Tensor::create<float>({3}, {0.1f, 0.2f, 0.3f});
  Tensor output = Tensor::create<float>({3}, {0.3006f, 0.3322f, 0.3672f});
  CATCH_REQUIRE(F::allClose(F::softmax(input), output));

  constexpr float inf = std::numeric_limits<float>::infinity();
  input = Tensor::create<float>({3}, {0.1f, 0.2f, -inf});
  output = Tensor::create<float>({3}, {0.4750f, 0.5250f, 0.0f});
  CATCH_REQUIRE(F::allClose(F::softmax(input), output));
}

CATCH_TEST_CASE("test rand q4", "[ly][operators][cpu][rand]") {
  constexpr uint64_t seed = 12345;

  lut::Random r(seed);
  Tensor x = F::rand({128, 32}, DType::kFloat, Device::getCpu(), &r);
  x = F::cast(x, DType::kQInt4Group32);
  x = F::cast(x, DType::kFloat);

  CATCH_REQUIRE(F::allClose(op::cpu::fingerprint(x), Tensor::create<float>({8}, {
    -0.3929, -0.3768, -1.0137, 0.8755, -0.9678, -0.8955, 0.0, 0.2485
  })));
}

}  // cpu
}  // op
}  // ly
