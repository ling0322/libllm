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

#pragma once

#include "libllm/cpu/kernel/abstract.h"
#include "libllm/cpu/kernel/block.h"
#include "libllm/cpu/kernel/cvt.h"
#include "libllm/cpu/kernel/gemv.h"
#include "libllm/lut/log.h"
#include "libllm/lut/time.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

template<int MC, int KC, int NC, int MR, int NR, typename T, CpuMathBackend TYPE, Mode MODE>
class Gemm {
 public:
  Gemm() {
    int packedSize = (MC * KC + KC * NC) * sizeof(T);
    _packedBuffer = (T *)malloc(packedSize);

    T *A = _packedBuffer;
    T *B = A + MC * KC;

    _bufferA = Block<T>{A, MR, (MC / MR) * KC, MR, false};
    _bufferB = Block<T>{B, NR, (NC / NR) * KC, NR, false};
  }

  ~Gemm() {
    free(_packedBuffer);
    _packedBuffer = nullptr;
  }

  void apply(const GemmArgs<T, T, T> &args) {
    _inputA = Block<T>{(T *)args.A, args.lda, args.M, args.K, args.transA};
    _inputB = Block<T>{(T *)args.B, args.ldb, args.K, args.N, args.transB};
    _inputC = Block<T>{(T *)args.C, args.ldc, args.M, args.N, false};

    split0ByNC();
  }

 private:
  T *_packedBuffer;

  Block<T> _bufferA;
  Block<T> _bufferB;

  Block<T> _inputA;
  Block<T> _inputB;
  Block<T> _inputC;

  void split0ByNC() {
    int nb = _inputB.numCols / NC;
    int nc = _inputB.numCols % NC;

    for (int i = 0; i < nb; ++i) {
      Block<T> Bn = _inputB.sliceCol(i * NC, NC);
      Block<T> Cj = _inputC.sliceCol(i * NC, NC);
      split1ByKC(Bn, Cj);
    }

    if (nc) {
      Block<T> Bn = _inputB.sliceCol(nb * NC, nc);
      Block<T> Cj = _inputC.sliceCol(nb * NC, nc);
      split1ByKC(Bn, Cj);
    }
  }

  void split1ByKC(Block<T> Bn, Block<T> Cj) {
    int kb = Bn.numRows / KC;
    int kc = Bn.numRows % KC;

    for (int i = 0; i < kb; ++i) {
      Block<T> Bkn = Bn.sliceRow(i * KC, KC);
      Block<T> Ak = _inputA.sliceCol(i * KC, KC);
      PackedBlock<T> Bp = Pack<T, MODE>(Bkn, _bufferB, NR);
      split2ByMC(Ak, Bp, Cj);
    }

    if (kc) {
      Block<T> Bkn = Bn.sliceRow(kb * KC, kc);
      Block<T> Ak = _inputA.sliceCol(kb * KC, kc);
      PackedBlock<T> Bp = Pack<T, MODE>(Bkn, _bufferB, NR);
      split2ByMC(Ak, Bp, Cj);
    }
  }

  void split2ByMC(Block<T> Ak, PackedBlock<T> Bp, Block<T> Cj) {
    int mb = Ak.numRows / MC;
    int mc = Ak.numRows % MC;

    for (int i = 0; i < mb; ++i) {
      Block<T> Amk = Ak.sliceRow(i * MC, MC);
      Block<T> Cij = Cj.sliceRow(i * MC, MC);
      PackedBlock<T> Ap = Pack<T, MODE>(Amk.t(), _bufferA, MR);
      macroKernel(Ap, Bp, Cij);
    }

    if (mc) {
      Block<T> Amk = Ak.sliceRow(mb * MC, mc);
      Block<T> Cij = Cj.sliceRow(mb * MC, mc);
      PackedBlock<T> Ap = Pack<T, MODE>(Amk.t(), _bufferA, MR);
      macroKernel(Ap, Bp, Cij);
    }
  }

  // GEMM macro-kernel: A(packed: MC, KC) DOT B(packed: KC, NC) -> C(MC, NC)
  void macroKernel(PackedBlock<T> A, PackedBlock<T> B, Block<T> C) {
    int np = (C.numCols + NR - 1) / NR;
    int mp = (C.numRows + MR - 1) / MR;
    int lastNr = C.numCols % NR;
    int lastMr = C.numRows % MR;

#pragma omp parallel for if (MODE == Mode::OMP)
    for (int i = 0; i < np; ++i) {
      for (int j = 0; j < mp; ++j) {
        int nr = (i != np - 1 || lastNr == 0) ? NR : lastNr;
        int mr = (j != mp - 1 || lastMr == 0) ? MR : lastMr;

        Block<T> Aj = A.block(j);
        Block<T> Bi = B.block(i);
        Block<T> Cji = C.slice(j * MR, i * NR, mr, nr);

        microKernel(Aj, Bi, Cji);
      }
    }
  }

  void microKernel(Block<T> A, Block<T> B, Block<T> C) {
    T dataCb[MR * NR];

    if (C.numRows < MR || C.numCols < NR) {
      Block<T> Cb{dataCb, NR, MR, NR, false};
      Cb.fillZero();

      Block<T> Cbs = Cb.slice(0, 0, C.numRows, C.numCols);
      C.copyTo(Cbs);

      gemmKernel<T, T, T, MR, NR, TYPE>(A.numRows, A.data, B.data, Cb.data, Cb.stride);
      Cbs.copyTo(C);
    } else {
      gemmKernel<T, T, T, MR, NR, TYPE>(A.numRows, A.data, B.data, C.data, C.stride);
    }
  }
};

/// @brief Provides GEMM interface with dispatcher for GEMM/GEMV.
template<int MC, int KC, int NC, int MR, int NR, typename T, CpuMathBackend TYPE, Mode MODE>
void gemm(const GemmArgs<T, T, T> &args) {
  if (args.M == 1) {
    std::fill(args.C, args.C + args.N, 0.0f);

    gemv<T, T, T, TYPE, MODE>(GemvArgs<T, T, T>{
        !args.transB,
        args.transB ? args.N : args.K,
        args.transB ? args.K : args.N,
        args.B,
        args.ldb,
        args.A,
        args.transA ? args.lda : 1,
        args.C,
        1});
  } else if (args.N == 1) {
    bool needPackC = args.ldc != 1;
    if (args.ldc != 1) {
      NOT_IMPL();
    } else {
      std::fill(args.C, args.C + args.M, 0.0f);
    }

    gemv<T, T, T, TYPE, MODE>(GemvArgs<T, T, T>{
        args.transA,
        args.transA ? args.K : args.M,
        args.transA ? args.M : args.K,
        args.A,
        args.lda,
        args.B,
        args.transB ? 1 : args.ldb,
        args.C,
        args.ldc});
  } else {
    Gemm<MC, KC, NC, MR, NR, T, TYPE, MODE>().apply(args);
  }
}

/// @brief Provides quantized GEMM interface with dispatcher for GEMM/GEMV.
template<
    int MC,
    int KC,
    int NC,
    int MR,
    int NR,
    typename T,
    typename TQ,
    CpuMathBackend TYPE,
    Mode MODE>
void qgemm(const GemmArgs<T, TQ, T> &args) {
  if (args.M == 1) {
    // fill C with zero.
    std::fill(args.C, args.C + args.N, 0.0f);

    gemv<TQ, T, T, TYPE, MODE>(GemvArgs<TQ, T, T>{
        !args.transB,
        args.transB ? args.N : args.K,
        args.transB ? args.K : args.N,
        args.B,
        args.ldb,
        args.A,
        args.transA ? args.lda : 1,
        args.C,
        1});
  } else {
    int numelB = args.K * args.N;
    lut::c_ptr<T> B = alignedAlloc<T>(numelB);
    cvt<TQ, T, TYPE, MODE>(numelB, args.B, 0, B.get());

    int ldb = args.transB ? args.K : args.N;

    GemmArgs<T, T, T> gemmArgs{
        args.transA,
        args.transB,
        args.M,
        args.N,
        args.K,
        args.A,
        args.lda,
        B.get(),
        ldb,
        args.C,
        args.ldc};
    Gemm<MC, KC, NC, MR, NR, T, TYPE, MODE>().apply(gemmArgs);
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
