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
#include "libllm/lut/time.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

// block is a sub area of a matrix.
struct Block {
  float *data;
  int32_t stride;
  int32_t numRows;
  int32_t numCols;
  bool transposed;

  constexpr Block sliceRow(int row, int nr);
  constexpr Block sliceCol(int col, int nc);
  constexpr Block slice(int row, int col, int nr, int nc);
  inline void copyTo(Block tgt);
  constexpr Block T();
  constexpr void fillZero();
};

struct PackedBlock {
  float *data;
  int32_t packSize;
  int32_t numRows;
  int32_t numBlocks;

  constexpr Block block(int i);
};


template<int MC, int KC, int NC, class TKernel, Mode MODE>
class GEMMCommon {
 public:
  GEMMCommon();
  ~GEMMCommon();

  static constexpr int MR = TKernel::MR;
  static constexpr int NR = TKernel::NR;

  // Compute C <- A * B
  void Apply(
      bool TransA, bool TransB,
      int M, int N, int K,
      const float *A, int lda,
      const float *B, int ldb,
      float *C, int ldc);

 private:
  float *_packedBuffer;

  Block _bufferA;
  Block _bufferB;

  Block _inputA;
  Block _inputB;
  Block _inputC;

  void split0ByNC();
  void split1ByKC(Block Bn, Block Cj);
  void split2ByMC(Block Ak, PackedBlock Bp, Block Cj);
};

template<Mode MODE>
PackedBlock Pack(Block src, Block buf, int pack_size) {
  int numBlock = src.numCols / pack_size;
  int kc = src.numRows;
  PackedBlock tgt { buf.data, pack_size, kc, numBlock };
  CHECK(pack_size * numBlock * kc <= buf.numCols * buf.numRows);

  #pragma omp parallel for if(MODE == Mode::OMP)
  for (int b = 0; b < numBlock; ++b) {
    Block srcBlock = src.sliceCol(b * pack_size, pack_size);
    Block tgtBlock = tgt.block(b);
    srcBlock.copyTo(tgtBlock);
  }

  int nc = src.numCols % pack_size;
  if (nc) {
    Block srcBlock = src.sliceCol(numBlock * pack_size, nc);
    Block tgtBlock = tgt.block(numBlock);
    tgtBlock.fillZero();

    tgtBlock = tgtBlock.sliceCol(0, nc);
    srcBlock.copyTo(tgtBlock);
    ++tgt.numBlocks;
  }

  return tgt;
}

// -- class Block ----------

constexpr Block Block::sliceRow(int row, int nr) {
  return slice(row, 0, nr, numCols);
}
constexpr Block Block::sliceCol(int col, int nc) {
  return slice(0, col, numRows, nc);
}
constexpr Block Block::slice(int row, int col, int nr, int nc) {
  return Block {
    data + (transposed ? row + col * stride : row * stride + col),
    stride,
    nr,
    nc,
    transposed
  };
}

inline void Block::copyTo(Block tgt) {
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

constexpr Block Block::T() {
  return Block {
    data,
    stride,
    numCols,
    numRows,
    !transposed
  };
}
constexpr void Block::fillZero() {
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
constexpr Block PackedBlock::block(int i) {
  return Block {
    data + packSize * numRows * i,
    packSize,
    numRows,
    packSize,
    false
  };
}

// -- class GEMMCommon ----------

template<int MC, int KC, int NC, class TKernel, Mode MODE>
inline GEMMCommon<MC, KC, NC, TKernel, MODE>::GEMMCommon() {
  int packedSize = (MC * KC + KC * NC) * sizeof(float);
  _packedBuffer = (float *)malloc(packedSize);

  float *A = _packedBuffer;
  float *B = A + MC * KC;

  _bufferA = Block { A, MR, (MC / MR) * KC, MR, false };
  _bufferB = Block { B, NR, (NC / NR) * KC, NR, false };
}

template<int MC, int KC, int NC, class TKernel, Mode MODE>
inline GEMMCommon<MC, KC, NC, TKernel, MODE>::~GEMMCommon() {
  free(_packedBuffer);
  _packedBuffer = nullptr;
}

template<int MC, int KC, int NC, class TKernel, Mode MODE>
inline void GEMMCommon<MC, KC, NC, TKernel, MODE>::split0ByNC() {
  int nb = _inputB.numCols / NC;
  int nc = _inputB.numCols % NC;

  for (int i = 0; i < nb; ++i) {
    Block Bn = _inputB.sliceCol(i * NC, NC);
    Block Cj = _inputC.sliceCol(i * NC, NC);
    split1ByKC(Bn, Cj);
  }

  if (nc) {
    Block Bn = _inputB.sliceCol(nb * NC, nc);
    Block Cj = _inputC.sliceCol(nb * NC, nc);
    split1ByKC(Bn, Cj);
  }
}

template<int MC, int KC, int NC, class TKernel, Mode MODE>
inline void GEMMCommon<MC, KC, NC, TKernel, MODE>::split1ByKC(Block Bn, Block Cj) {
  int kb = Bn.numRows / KC;
  int kc = Bn.numRows % KC;

  for (int i = 0; i < kb; ++i) {
    Block Bkn = Bn.sliceRow(i * KC, KC);
    Block Ak = _inputA.sliceCol(i * KC, KC);
    PackedBlock Bp = Pack<MODE>(Bkn, _bufferB, NR);
    split2ByMC(Ak, Bp, Cj);
  }

  if (kc) {
    Block Bkn = Bn.sliceRow(kb * KC, kc);
    Block Ak = _inputA.sliceCol(kb * KC, kc);
    PackedBlock Bp = Pack<MODE>(Bkn, _bufferB, NR);
    split2ByMC(Ak, Bp, Cj);
  }
}


template<class TMicroKernel>
void callGemmMicroKernel(Block A, Block B, Block C) {
  constexpr int MR = TMicroKernel::MR;
  constexpr int NR = TMicroKernel::NR;
  float dataCb[MR * NR];

  if (C.numRows < MR || C.numCols < NR) {
    Block Cb{dataCb, NR, MR, NR, false};
    Cb.fillZero();

    Block Cbs = Cb.slice(0, 0, C.numRows, C.numCols);
    C.copyTo(Cbs);

    TMicroKernel::apply(A.numRows, A.data, B.data, Cb.data, Cb.stride);
    Cbs.copyTo(C);
  } else {
    TMicroKernel::apply(A.numRows, A.data, B.data, C.data, C.stride);
  }
}

// GEMM macro-kernel: A(packed: MC, KC) DOT B(packed: KC, NC) -> C(MC, NC)
template<int MC, int KC, int NC, class TMicroKernel, Mode MODE>
void applyGemmMacroKernel(PackedBlock A, PackedBlock B, Block C) {
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

      Block Aj = A.block(j);
      Block Bi = B.block(i);
      Block Cji = C.slice(j * MR, i * NR, mr, nr);

      callGemmMicroKernel<TMicroKernel>(Aj, Bi, Cji);
    }
  }
}


template<int MC, int KC, int NC, class TKernel, Mode MODE>
inline void GEMMCommon<MC, KC, NC, TKernel, MODE>::split2ByMC(Block Ak, PackedBlock Bp, Block Cj) {
  int mb = Ak.numRows / MC;
  int mc = Ak.numRows % MC;

  for (int i = 0; i < mb; ++i) {
    Block Amk = Ak.sliceRow(i * MC, MC);
    Block Cij = Cj.sliceRow(i * MC, MC);
    PackedBlock Ap = Pack<MODE>(Amk.T(), _bufferA, MR);
    applyGemmMacroKernel<MC, KC, NC, TKernel, MODE>(Ap, Bp, Cij);
  }

  if (mc) {
    Block Amk = Ak.sliceRow(mb * MC, mc);
    Block Cij = Cj.sliceRow(mb * MC, mc);

    PackedBlock Ap = Pack<MODE>(Amk.T(), _bufferA, MR);
    applyGemmMacroKernel<MC, KC, NC, TKernel, MODE>(Ap, Bp, Cij); 
  }
}

template<int MC, int KC, int NC, class TKernel, Mode MODE>
inline void GEMMCommon<MC, KC, NC, TKernel, MODE>::Apply(
    bool transa, bool transb,
    int m, int n, int k,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc) {
  _inputA = Block { (float *)A, lda, m, k, transa };
  _inputB = Block { (float *)B, ldb, k, n, transb };
  _inputC = Block { (float *)C, ldc, m, n, false };

  split0ByNC();
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
