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

// The input of LLM.
type Prompt interface {
	AppendText(text string)
	AppendControlToken(text string)
	AppendAudio(data []byte, format AudioFormat)

	// Update the llmPrompt_t instance according to the current prompt.
	updatePromptHandle(handle *promptHandle) error
}

type promptImpl struct {
	elements []promptElem
}

type promptElem interface {
	AppendTo(handle *promptHandle) error
}

type textPromptElem struct {
	text string
}

type controlTokenPromptElem struct {
	token string
}

type audioPromptElem struct {
	payload []byte
	format  AudioFormat
}

type promptHandle struct {
	handle *C.llmPrompt_t
}

func NewPrompt() Prompt {
	return &promptImpl{}
}

func (p *promptImpl) AppendText(text string) {
	p.elements = append(p.elements, &textPromptElem{text})
}

func (p *promptImpl) AppendControlToken(text string) {
	p.elements = append(p.elements, &controlTokenPromptElem{text})
}

func (p *promptImpl) AppendAudio(audio []byte, format AudioFormat) {
	p.elements = append(p.elements, &audioPromptElem{audio, format})
}

func (p *promptImpl) updatePromptHandle(handle *promptHandle) error {
	if len(p.elements) == 0 {
		return errors.New("prompt is empty")
	}

	for _, elem := range p.elements {
		err := elem.AppendTo(handle)
		if err != nil {
			return err
		}
	}

	return nil
}

func (e *textPromptElem) AppendTo(handle *promptHandle) error {
	cText := C.CString(e.text)
	defer C.free(unsafe.Pointer(cText))

	status := C.llmPrompt_AppendText(handle.handle, cText)
	if status != C.LLM_OK {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return nil
}

func (e *controlTokenPromptElem) AppendTo(handle *promptHandle) error {
	cToken := C.CString(e.token)
	defer C.free(unsafe.Pointer(cToken))

	status := C.llmPrompt_AppendControlToken(handle.handle, cToken)
	if status != C.LLM_OK {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return nil
}

func (e *audioPromptElem) AppendTo(handle *promptHandle) error {
	cPayload := C.CBytes(e.payload)
	defer C.free(unsafe.Pointer(cPayload))

	status := C.llmPrompt_AppendAudio(
		handle.handle,
		(*C.llmByte_t)(cPayload),
		C.int64_t(len(e.payload)),
		C.int32_t(Pcm16kHz16BitMono))
	if status != C.LLM_OK {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return nil
}

func newPromptHandle() (*promptHandle, error) {
	cHandle := C.llmPrompt_New()
	if cHandle == nil {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	handle := &promptHandle{
		cHandle,
	}
	runtime.SetFinalizer(handle, func(h *promptHandle) {
		status := C.llmPrompt_Delete(h.handle)
		if status != C.LLM_OK {
			fmt.Fprintln(os.Stderr, "failed to call llmPrompt_Delete()")
		}
	})

	return handle, nil
}
