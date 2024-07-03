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
	"unsafe"
)

// A LLM.
type Model interface {
	GetName() string
	Complete(config CompletionConfig, prompt Prompt) (Completion, error)
}

type modelHandle struct {
	handle *C.llmModel_t
}

type modelImpl struct {
	handle *modelHandle
}

// Load a LLM model from `modelPath`, then save it to the specified device.
func NewModel(modelPath string, device Device) (Model, error) {
	err := initLlm()
	if err != nil {
		return nil, err
	}

	handle, err := newModelHandle()
	if err != nil {
		return nil, err
	}

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if C.llmModel_SetFile(handle.handle, cPath) != C.LLM_OK {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	if C.llmModel_SetDevice(handle.handle, C.int32_t(device)) != C.LLM_OK {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	if C.llmModel_Load(handle.handle) != C.LLM_OK {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	model := &modelImpl{
		handle: handle,
	}
	return model, nil
}

func (m *modelImpl) Complete(config CompletionConfig, prompt Prompt) (Completion, error) {
	comp, err := newCompletionImpl(m.handle)
	if err != nil {
		return nil, err
	}

	err = config.updateCompHandle(comp.handle)
	if err != nil {
		return nil, err
	}

	promptHandle, err := newPromptHandle()
	if err != nil {
		return nil, err
	}

	err = prompt.updatePromptHandle(promptHandle)
	if err != nil {
		return nil, err
	}

	ok := C.llmCompletion_SetPrompt(comp.handle.handle, promptHandle.handle)
	if ok != C.LLM_OK {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return comp, nil
}

// Get the name of model.
func (m *modelImpl) GetName() string {
	name := C.llmModel_GetName(m.handle.handle)
	if name == nil {
		return ""
	} else {
		return C.GoString(name)
	}
}

func newModelHandle() (*modelHandle, error) {
	cHandle := C.llmModel_New()
	if cHandle == nil {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	handle := &modelHandle{
		cHandle,
	}
	runtime.SetFinalizer(handle, func(h *modelHandle) {
		status := C.llmModel_Delete(h.handle)
		if status != C.LLM_OK {
			fmt.Fprintln(os.Stderr, "failed to call llmModel_Delete()")
		}
	})

	return handle, nil
}
