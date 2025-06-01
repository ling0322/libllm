// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "lten/cpu/kernel/abstract.h"
#include "lten/mp.h"
#include "lutil/log.h"
#include "lutil/time.h"

namespace lten {
namespace op {
namespace cpu {
namespace kernel {

// block is a sub area of a matrix. Used in implementation of gemm.
template<typename T>
struct Block {
  T *data;
  int32_t stride;
  int32_t numRows;
  int32_t numCols;
  bool transposed;

  constexpr Block<T> sliceRow(int row, int nr) const;
  constexpr Block<T> sliceCol(int col, int nc) const;
  constexpr Block<T> slice(int row, int col, int nr, int nc) const;
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

  constexpr Block<T> block(int i) const;
};

template<typename T, Mode MODE>
PackedBlock<T> Pack(Block<T> src, Block<T> buf, int pack_size) {
  int numBlock = src.numCols / pack_size;
  int kc = src.numRows;
  PackedBlock<T> tgt{buf.data, pack_size, kc, numBlock};
  CHECK(pack_size * numBlock * kc <= buf.numCols * buf.numRows);

  auto closure = [src, tgt, pack_size](MP::Context ctx) {
    int b = ctx.getBlockIdx();
    Block<T> srcBlock = src.sliceCol(b * pack_size, pack_size);
    Block<T> tgtBlock = tgt.block(b);
    srcBlock.copyTo(tgtBlock);
  };

  if (MODE == Mode::OMP) {
    MP::parallelFor(numBlock, closure);
  } else {
    for (int i = 0; i < numBlock; ++i) {
      closure(MP::Context(i, numBlock, 0));
    }
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

template<typename T>
constexpr Block<T> Block<T>::sliceRow(int row, int nr) const {
  return slice(row, 0, nr, numCols);
}
template<typename T>
constexpr Block<T> Block<T>::sliceCol(int col, int nc) const {
  return slice(0, col, numRows, nc);
}

template<typename T>
constexpr Block<T> Block<T>::slice(int row, int col, int nr, int nc) const {
  return Block{
      data + (transposed ? row + col * stride : row * stride + col),
      stride,
      nr,
      nc,
      transposed};
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
  return Block<T>{data, stride, numCols, numRows, !transposed};
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
constexpr Block<T> PackedBlock<T>::block(int i) const {
  return Block<T>{data + packSize * numRows * i, packSize, numRows, packSize, false};
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace lten
