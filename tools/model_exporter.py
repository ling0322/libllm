# The MIT License (MIT)
#
# Copyright (c) 2023 Xiaoyang Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import nn

import struct
import torch
import math
import numpy as np
from enum import Enum
import torch.nn.functional as F

DTYPE_UNKNOWN = 0
DTYPE_FP32 = 1
DTYPE_INT64 = 2
DTYPE_UINT8 = 3
DTYPE_FP16 = 4
DTYPE_Q4 = 5
DTYPE_INT8 = 6

class Quant(Enum):
    NONE = 0
    Q4 = 2

    @classmethod
    def parse(cls, quant: str) -> Quant:
        quant = quant.lower()
        if quant == "q4":
            return Quant.Q4
        elif quant == "none":
            return Quant.NONE
        else:
            raise NotImplementedError("unsupported quantization type: " + quant)

class Context:
    """stores the context of  module and tensor."""

    def __init__(self, name="", quant=Quant.NONE) -> None:
        self._ns = name
        self._quant = quant

    def _copy(self) -> Context:
        """get a cpy of current context."""
        ctx = Context()
        ctx._ns = self._ns
        ctx._quant = self._quant

        return ctx

    def _subname(self, name: str) -> str:
        return name if not self._ns else self._ns + '.' + name

    @property
    def name(self) -> str:
        return self._ns if self._ns else "<root>"
    
    @property
    def quant(self) -> Quant:
        return self._quant

    def with_subname(self, name: str) -> Context:
        """get the context object with a sub-namespace"""
        ctx = self._copy()
        ctx._ns = self._subname(name)
        return ctx

    def with_quant(self, quant: Quant) -> Context:
        """returns a context object the same as current context, the only difference is
        quantization setting."""
        ctx = self._copy()
        ctx._quant = quant
        return ctx

class Quantization:
    @classmethod
    def _pack_uint8_to_uint4x2(cls, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dtype == torch.uint8 and tensor.dim() == 1
        assert torch.all(tensor <= 15)

        if tensor.shape[0] % 2 == 1:
            pad_value = torch.zeros((1, ), dtype=torch.uint8, device=tensor.device)
            tensor = torch.cat((tensor, pad_value))
        tensor = tensor.reshape(-1, 2)
        tensor = tensor[:, 0].type(torch.uint8) + tensor[:, 1].type(torch.uint8) * 16
        return tensor

    @classmethod
    def quantize_to_q4(cls, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """1D tensor to qdata (q4x2), scale (fp16), zero_point (int8) """
        grouped_data = tensor.reshape(-1, 32)
        num_group = grouped_data.shape[0]

        min_value = torch.min(grouped_data, 1).values
        max_value = torch.max(grouped_data, 1).values

        # zero point is in range(0, 15), here to make sure the zero point will not overflow.
        min_value[min_value > 0] = 0
        max_value[max_value < 0] = 0

        group_scale = (max_value - min_value) / 15
        group_zero_point = torch.round(-min_value / group_scale).type(torch.int32)
        assert torch.all(group_zero_point <= 15) and torch.all(group_zero_point >= 0)
        group_zero_point = group_zero_point.type(torch.uint8)
        group_zero_point = cls._pack_uint8_to_uint4x2(group_zero_point)

        qdata = (grouped_data - min_value.reshape(num_group, 1)) / group_scale.reshape(num_group, 1)
        qdata = torch.round(qdata).reshape(-1).type(torch.int32)
        assert torch.all(qdata <= 15) and torch.all(qdata >= 0)
        qdata = qdata.type(torch.uint8)
        qdata = cls._pack_uint8_to_uint4x2(qdata)

        return qdata, group_scale.type(torch.float16), group_zero_point

class TensorWriter:
    """write tensor to file with llyn tensor format."""

    def __init__(self, filename: str) -> None:
        self._fp = open(filename, 'wb')
        self._fp.write(b"llyn::tdic      ")

        # reserved
        self._fp.write(struct.pack('<i', 0))
        self._fp.write(struct.pack('<i', 0))

        # numTensors, this field will be filled when writer is closing.
        self._num_tensors = 0
        self._fp.write(struct.pack('<i', self._num_tensors))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print("__exit__")
        self._fp.write(struct.pack('<h', 0x55aa))
        self._fp.seek(24)
        self._fp.write(struct.pack('<i', self._num_tensors))
        self._fp.close()

    def _write_tensor_elem(self, tensor: torch.Tensor, dtype=DTYPE_UNKNOWN):
        if dtype == DTYPE_UNKNOWN:
            dtype = self._dtype_to_libllm_dtype(tensor.dtype)

        numel = tensor.numel()
        if dtype in {DTYPE_Q4}:
            # in q4, two int4 merged into a uint8 byte.
            numel *= 2

        self._fp.write(struct.pack('<h', dtype))
        self._fp.write(struct.pack('<q', numel))

        np_tensor = tensor.cpu().detach().contiguous().numpy()
        assert np_tensor.dtype in {np.dtype(np.float32), np.dtype(np.float16), np.dtype(np.int64), np.dtype(np.int8), np.dtype(np.uint8)}

        self._fp.write(np_tensor.tobytes())
        self._fp.write(struct.pack('<h', 0x55aa))

    def _write_tensor_header(self, shape, num_slot=1):
        self._fp.write(b'tnsr')
        self._fp.write(struct.pack('<h', len(shape)))
        for size in shape:
            self._fp.write(struct.pack('<i', size))
        
        # header for data.
        self._fp.write(b"tdat")
        self._fp.write(struct.pack('<i', num_slot))


    def _write_tensor(self, tensor: torch.Tensor):
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        assert tensor.dtype in {torch.float32, torch.int64}

        self._write_tensor_header(tensor.shape)
        self._write_tensor_elem(tensor)
        

    def _dtype_to_libllm_dtype(self, dtype):
        if dtype == torch.float32:
            return DTYPE_FP32
        if dtype == torch.float16:
            return DTYPE_FP16
        if dtype == torch.int64:
            return DTYPE_INT64
        if dtype == torch.int8:
            return DTYPE_INT8
        if dtype == torch.uint8:
            return DTYPE_UINT8
        else:
            raise Exception("dtype not supported")

    def _write_tensor_q4(self, tensor: torch.Tensor):
        """ quantize the pytorch tensor to q4 format (int4 asymmetric quantization, group size 32,
        scale format float16). Then write to self._fp.
        """
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        assert tensor.dtype in {torch.float32, torch.int64}

        self._write_tensor_header(tensor.shape, num_slot=3)
        
        qdata, scale, zero_point = Quantization.quantize_to_q4(tensor)
        self._write_tensor_elem(qdata, DTYPE_Q4)
        self._write_tensor_elem(scale)
        self._write_tensor_elem(zero_point)

    def write_tensor(self, ctx: Context, tensor: torch.Tensor):
        print(f"write tensor {ctx.name}, shape={tensor.shape}, quant={ctx.quant}")
        self._num_tensors += 1

        if len(ctx.name) > 1024:
            raise Exception('name too long')
        name = ctx.name.encode('utf-8')
        self._fp.write(struct.pack('<h', len(name)))
        self._fp.write(name)

        if ctx.quant == Quant.NONE:
            self._write_tensor(tensor)
        elif ctx.quant == Quant.Q4:
            self._write_tensor_q4(tensor)
        else:
            raise NotImplementedError(ctx.quant)

class ModelExporter:
    def __init__(self, writer: TensorWriter) -> None:
        self._writer = writer

    def _write(self, ctx: Context, tensor: torch.Tensor):
        self._writer.write_tensor(ctx, tensor)

    def export_embedding(self, ctx: Context, module: nn.Embedding):
        self._write(ctx.with_subname("weight"), module.weight)

    def export_layer_norm(self, ctx: Context, module: nn.LayerNorm):
        self._write(ctx.with_subname("weight").with_quant(Quant.NONE), module.weight)
        self._write(ctx.with_subname("bias").with_quant(Quant.NONE), module.bias)
