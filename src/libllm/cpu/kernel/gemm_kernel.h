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

#include "libllm/cpu/kernel/common.h"
#include "libllm/lut/log.h"
#include "libllm/lut/time.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

// block is a sub area of a matrix.
template<typename T>
struct Block {
  T *data;
  int32_t stride;
  int32_t numRows;
  int32_t numCols;
  bool transposed;

  constexpr Block<T> sliceRow(int row, int nr);
  constexpr Block<T> sliceCol(int col, int nc);
  constexpr Block<T> slice(int row, int col, int nr, int nc);
  inline void copyTo(Block<T> tgt);
  constexpr Block<T> t();
  constexpr void fillZero();
};

template<typename T>
struct PackedBlock {
  T *data;
  int32_t packSize;
  int32_t numRows;
  int32_t numBlocks;

  constexpr Block<T> block(int i);
};


template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
class GemmKernel {
 public:
  GemmKernel();
  ~GemmKernel();

  static constexpr int MR = TMicroKernel::MR;
  static constexpr int NR = TMicroKernel::NR;

  // Compute C <- A * B
  void Apply(
      bool TransA, bool TransB,
      int M, int N, int K,
      const T *A, int lda,
      const T *B, int ldb,
      T *C, int ldc);

 private:
  T *_packedBuffer;

  Block<T> _bufferA;
  Block<T> _bufferB;

  Block<T> _inputA;
  Block<T> _inputB;
  Block<T> _inputC;

  void split0ByNC();
  void split1ByKC(Block<T> Bn, Block<T> Cj);
  void split2ByMC(Block<T> Ak, PackedBlock<T> Bp, Block<T> Cj);
};

template<typename T, Mode MODE>
PackedBlock<T> Pack(Block<T> src, Block<T> buf, int pack_size) {
  int numBlock = src.numCols / pack_size;
  int kc = src.numRows;
  PackedBlock<T> tgt { buf.data, pack_size, kc, numBlock };
  CHECK(pack_size * numBlock * kc <= buf.numCols * buf.numRows);

  #pragma omp parallel for if(MODE == Mode::OMP)
  for (int b = 0; b < numBlock; ++b) {
    Block<T> srcBlock = src.sliceCol(b * pack_size, pack_size);
    Block<T> tgtBlock = tgt.block(b);
    srcBlock.copyTo(tgtBlock);
  }

  int nc = src.numCols % pack_size;
  if (nc) {
    Block<T> srcBlock = src.sliceCol(numBlock * pack_size, nc);
    Block<T> tgtBlock = tgt.block(numBlock);
    tgtBlock.fillZero();

    tgtBlock = tgtBlock.sliceCol(0, nc);
    srcBlock.copyTo(tgtBlock);
    ++tgt.numBlocks;
  }

  return tgt;
}

// -- class Block ----------

template<typename T>
constexpr Block<T> Block<T>::sliceRow(int row, int nr) {
  return slice(row, 0, nr, numCols);
}
template<typename T>
constexpr Block<T> Block<T>::sliceCol(int col, int nc) {
  return slice(0, col, numRows, nc);
}

template<typename T>
constexpr Block<T> Block<T>::slice(int row, int col, int nr, int nc) {
  return Block {
    data + (transposed ? row + col * stride : row * stride + col),
    stride,
    nr,
    nc,
    transposed
  };
}

template<typename T>
inline void Block<T>::copyTo(Block<T> tgt) {
  CHECK(numRows == tgt.numRows);
  CHECK(numCols == tgt.numCols);

  if ((!transposed) && (!tgt.transposed)) {
    for (int r = 0; r < numRows; ++r) {
      int tgtOffset = r * tgt.stride;
      int srcOffset = r * stride;
      for (int c = 0; c < numCols; ++c) {
        tgt.data[tgtOffset + c] = data[srcOffset + c];
      }
    }
  } else if (transposed && (!tgt.transposed)) {
    for (int r = 0; r < numRows; ++r) {
      int tgtOffset = r * tgt.stride;
      for (int c = 0; c < numCols; ++c) {
        tgt.data[tgtOffset + c] = data[r + c * stride];
      }
    }
  } else if ((!transposed) && tgt.transposed) {
    for (int r = 0; r < numRows; ++r) {
      int srcOffset = r * stride;
      for (int c = 0; c < numCols; ++c) {
        tgt.data[r + c * tgt.stride] = data[srcOffset + c];
      }
    }
  } else if (transposed && tgt.transposed) {
    for (int c = 0; c < numCols; ++c) {
      int srcOffset = c * stride;
      int tgtOffset = c * tgt.stride;
      for (int r = 0; r < numRows; ++r) {
          tgt.data[r + tgtOffset] = data[r + srcOffset];
      }
    }
  }
}

template<typename T>
constexpr Block<T> Block<T>::t() {
  return Block<T> {
    data,
    stride,
    numCols,
    numRows,
    !transposed
  };
}

template<typename T>
constexpr void Block<T>::fillZero() {
  for (int r = 0; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      if (transposed) {
        data[r + c * stride] = 0.0f;
      } else {
        data[r * stride + c] = 0.0f;
      }
    }
  }
}

template<typename T>
constexpr Block<T> PackedBlock<T>::block(int i) {
  return Block<T> {
    data + packSize * numRows * i,
    packSize,
    numRows,
    packSize,
    false
  };
}

// -- class GemmKernel ----------

template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
inline GemmKernel<MC, KC, NC, T, TMicroKernel, MODE>::GemmKernel() {
  int packedSize = (MC * KC + KC * NC) * sizeof(T);
  _packedBuffer = (T *)malloc(packedSize);

  T *A = _packedBuffer;
  T *B = A + MC * KC;

  _bufferA = Block<T> { A, MR, (MC / MR) * KC, MR, false };
  _bufferB = Block<T> { B, NR, (NC / NR) * KC, NR, false };
}

template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
inline GemmKernel<MC, KC, NC, T, TMicroKernel, MODE>::~GemmKernel() {
  free(_packedBuffer);
  _packedBuffer = nullptr;
}

template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
inline void GemmKernel<MC, KC, NC, T, TMicroKernel, MODE>::split0ByNC() {
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

template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
inline void GemmKernel<MC, KC, NC, T, TMicroKernel, MODE>::split1ByKC(Block<T> Bn, Block<T> Cj) {
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


template<typename T, class TMicroKernel>
void callGemmMicroKernel(Block<T> A, Block<T> B, Block<T> C) {
  constexpr int MR = TMicroKernel::MR;
  constexpr int NR = TMicroKernel::NR;
  T dataCb[MR * NR];

  if (C.numRows < MR || C.numCols < NR) {
    Block<T> Cb{dataCb, NR, MR, NR, false};
    Cb.fillZero();

    Block<T> Cbs = Cb.slice(0, 0, C.numRows, C.numCols);
    C.copyTo(Cbs);

    TMicroKernel::apply(A.numRows, A.data, B.data, Cb.data, Cb.stride);
    Cbs.copyTo(C);
  } else {
    TMicroKernel::apply(A.numRows, A.data, B.data, C.data, C.stride);
  }
}

// GEMM macro-kernel: A(packed: MC, KC) DOT B(packed: KC, NC) -> C(MC, NC)
template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
void applyGemmMacroKernel(PackedBlock<T> A, PackedBlock<T> B, Block<T> C) {
  constexpr int MR = TMicroKernel::MR;
  constexpr int NR = TMicroKernel::NR;

  int np = (C.numCols + NR - 1) / NR;
  int mp = (C.numRows + MR - 1) / MR;
  int lastNr = C.numCols % NR;
  int lastMr = C.numRows % MR;

  #pragma omp parallel for if(MODE == Mode::OMP)
  for (int i = 0; i < np; ++i) {
    for (int j = 0; j < mp; ++j) {
      int nr = (i != np - 1 || lastNr == 0) ? NR : lastNr;
      int mr = (j != mp - 1 || lastMr == 0) ? MR : lastMr;

      Block<T> Aj = A.block(j);
      Block<T> Bi = B.block(i);
      Block<T> Cji = C.slice(j * MR, i * NR, mr, nr);

      callGemmMicroKernel<T, TMicroKernel>(Aj, Bi, Cji);
    }
  }
}


template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
inline void GemmKernel<MC, KC, NC, T, TMicroKernel, MODE>::split2ByMC(
    Block<T> Ak,
    PackedBlock<T> Bp,
    Block<T> Cj) {
  int mb = Ak.numRows / MC;
  int mc = Ak.numRows % MC;

  for (int i = 0; i < mb; ++i) {
    Block<T> Amk = Ak.sliceRow(i * MC, MC);
    Block<T> Cij = Cj.sliceRow(i * MC, MC);
    PackedBlock<T> Ap = Pack<T, MODE>(Amk.t(), _bufferA, MR);
    applyGemmMacroKernel<MC, KC, NC, T, TMicroKernel, MODE>(Ap, Bp, Cij);
  }

  if (mc) {
    Block<T> Amk = Ak.sliceRow(mb * MC, mc);
    Block<T> Cij = Cj.sliceRow(mb * MC, mc);
    PackedBlock<T> Ap = Pack<T, MODE>(Amk.t(), _bufferA, MR);
    applyGemmMacroKernel<MC, KC, NC, T, TMicroKernel, MODE>(Ap, Bp, Cij); 
  }
}

template<int MC, int KC, int NC, typename T, class TMicroKernel, Mode MODE>
inline void GemmKernel<MC, KC, NC, T, TMicroKernel, MODE>::Apply(
    bool transa, bool transb,
    int m, int n, int k,
    const T *A, int lda,
    const T *B, int ldb,
    T *C, int ldc) {
  _inputA = Block<T> { (T *)A, lda, m, k, transa };
  _inputB = Block<T> { (T *)B, ldb, k, n, transb };
  _inputC = Block<T> { (T *)C, ldc, m, n, false };

  split0ByNC();
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
