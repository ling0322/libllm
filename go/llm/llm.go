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

// #cgo linux LDFLAGS: -ldl
// #cgo darwin LDFLAGS: -ldl
// #include <stdlib.h>
// #include "llm.h"
import "C"
import (
	"errors"
	"runtime"
)

type Model struct {
	model C.llm_model_t
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
}

type Completion struct {
	comp  C.llm_completion_t
	json  *llmJson
	err   error
	chunk Chunk
}

type Chunk struct {
	Text string `json:"text"`
}

func NewModel(filename, device string) (*Model, error) {
	err := initLlm()
	if err != nil {
		return nil, err
	}

	json := newJson()
	json.marshal(map[string]any{
		"filename": filename,
		"device":   device,
	})

	m := new(Model)
	C.llm_model_init(&m.model)
	runtime.SetFinalizer(m, func(m *Model) {
		C.llm_model_destroy(&m.model)
	})

	status := C.llm_model_load(&m.model, &json.json)
	if status != 0 {
		return nil, errors.New(C.GoString(C.llm_get_last_error_message()))
	}

	return m, nil
}

func DefaultCompletionConfig() CompletionConfig {
	config := CompletionConfig{}
	config.Temperature = 1.0
	config.TopK = 50
	config.TopP = 0.8

	return config
}

func (m *Model) Complete(history []Message, config CompletionConfig) (*Completion, error) {
	if config.Temperature <= 0 || config.TopK <= 0 || config.TopP <= 0 {
		return nil, errors.New("invalid completion config")
	}

	json := newJson()
	json.marshal(map[string]any{
		"temperature": config.Temperature,
		"top_k":       config.TopK,
		"top_p":       config.TopP,
		"messages":    history,
	})

	comp := new(Completion)
	comp.json = newJson()
	C.llm_completion_init(&comp.comp)
	runtime.SetFinalizer(comp, func(c *Completion) {
		C.llm_completion_destroy(&c.comp)
	})

	status := C.llm_model_complete(&m.model, &json.json, &comp.comp)
	if status != 0 {
		return nil, errors.New(C.GoString(C.llm_get_last_error_message()))
	}

	return comp, nil
}

func (c *Completion) Err() error {
	return c.err
}

func (c *Completion) Chunk() Chunk {
	return c.chunk
}

func (c *Completion) Next() bool {
	status := C.llm_completion_get_next_chunk(&c.comp, &c.json.json)
	if status == LLM_ERROR_EOF {
		return false
	} else if status != 0 {
		c.err = errors.New(C.GoString(C.llm_get_last_error_message()))
		return false
	}

	c.err = c.json.unmarshal(&c.chunk)
	return c.err == nil
}
