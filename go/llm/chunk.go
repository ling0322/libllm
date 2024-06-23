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

// Generate by Compeltion.
type Chunk struct {
	Text string
}

type chunkHandle struct {
	handle *C.llmChunk_t
}

func newChunkHandle() (*chunkHandle, error) {
	cHandle := C.llmChunk_New()
	if cHandle == nil {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	handle := &chunkHandle{
		cHandle,
	}
	runtime.SetFinalizer(handle, func(h *chunkHandle) {
		status := C.llmChunk_Delete(h.handle)
		if status != C.LLM_OK {
			fmt.Fprintln(os.Stderr, "failed to call llmPrompt_Delete()")
		}
	})

	return handle, nil
}
