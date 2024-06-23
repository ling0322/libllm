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

package llm

// #include <stdlib.h>
// #include "llm_api.h"
import "C"
import (
	"errors"
	"fmt"
	"os"
	"runtime"
)

// Config for LLM completion.
type Completion interface {
	IsActive() bool
	GenerateNextChunk() (Chunk, error)
}

type completionHandle struct {
	handle *C.llmCompletion_t
}

type completionImpl struct {
	handle      *completionHandle
	chunkHandle *chunkHandle
}

func (c *completionImpl) IsActive() bool {
	return C.llmCompletion_IsActive(c.handle.handle) != 0
}

func (c *completionImpl) GenerateNextChunk() (Chunk, error) {
	status := C.llmCompletion_GenerateNextChunk(c.handle.handle, c.chunkHandle.handle)
	if status != C.LLM_OK {
		return Chunk{}, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	chunk := Chunk{}
	cText := C.llmChunk_GetText(c.chunkHandle.handle)
	if cText == nil {
		return Chunk{}, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	chunk.Text = C.GoString(cText)
	return chunk, nil
}

func newCompletionImpl(modelHandle *modelHandle) (*completionImpl, error) {
	handle, err := newCompletionHandle(modelHandle)
	if err != nil {
		return nil, err
	}

	chunkHandle, err := newChunkHandle()
	if err != nil {
		return nil, err
	}

	return &completionImpl{
		handle:      handle,
		chunkHandle: chunkHandle,
	}, nil
}

func newCompletionHandle(m *modelHandle) (*completionHandle, error) {
	cHandle := C.llmCompletion_New(m.handle)
	if cHandle == nil {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	handle := &completionHandle{
		cHandle,
	}
	runtime.SetFinalizer(handle, func(h *completionHandle) {
		status := C.llmCompletion_Delete(h.handle)
		if status != C.LLM_OK {
			fmt.Fprintln(os.Stderr, "failed to call llmCompletion_Delete()")
		}
	})

	return handle, nil
}
