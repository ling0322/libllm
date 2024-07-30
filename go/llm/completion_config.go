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
	"unsafe"
)

// Config for LLM completion.
type CompletionConfig interface {
	SetTopP(topP float32)
	GetTopP() float32

	SetTopK(topK int)
	GetTopK() int

	SetTemperature(temperature float32)
	GetTemperature() float32

	SetConfig(key, value string)

	// update the llmCompletion_t according to the config.
	updateCompHandle(compHandle *completionHandle) error
}

type completionConfigImpl struct {
	topP        float32
	topK        int
	temperature float32

	kvConfig map[string]string
}

func NewCompletionConfig() CompletionConfig {
	return &completionConfigImpl{
		topP:        0.8,
		topK:        50,
		temperature: 1.0,
		kvConfig:    map[string]string{},
	}
}

func (c *completionConfigImpl) SetTopP(topP float32) {
	c.topP = topP
}

func (c *completionConfigImpl) GetTopP() float32 {
	return c.topP
}

func (c *completionConfigImpl) SetTopK(topK int) {
	c.topK = topK
}

func (c *completionConfigImpl) GetTopK() int {
	return c.topK
}

func (c *completionConfigImpl) SetTemperature(temperature float32) {
	c.temperature = temperature
}

func (c *completionConfigImpl) GetTemperature() float32 {
	return c.temperature
}

func (c *completionConfigImpl) SetConfig(key, value string) {
	c.kvConfig[key] = value
}

func (c *completionConfigImpl) updateCompHandle(compHandle *completionHandle) error {
	if C.llmCompletion_SetTopP(compHandle.handle, C.float(c.topP)) != C.LLM_OK {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	if C.llmCompletion_SetTopK(compHandle.handle, C.int(c.topK)) != C.LLM_OK {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	if C.llmCompletion_SetTemperature(compHandle.handle, C.float(c.temperature)) != C.LLM_OK {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	for key, value := range c.kvConfig {
		cKey := C.CString(key)
		cValue := C.CString(value)
		ok := C.llmCompletion_SetConfig(compHandle.handle, cKey, cValue)
		C.free(unsafe.Pointer(cKey))
		C.free(unsafe.Pointer(cValue))
		if ok != C.LLM_OK {
			return errors.New(C.GoString(C.llmGetLastErrorMessage()))
		}
	}

	return nil
}
