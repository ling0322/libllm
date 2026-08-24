// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package llm

// #cgo linux LDFLAGS: -ldl
// #cgo darwin LDFLAGS: -ldl
// #include <stdlib.h>
// #include "llm.h"
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

type Model struct {
	mutex  sync.Mutex
	engine C.llm_engine_t
	closed bool
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

type completionEvent struct {
	chunk    Chunk
	err      error
	hasChunk bool
}

type Completion struct {
	requestID string
	mutex     sync.Mutex
	ready     *sync.Cond
	events    []completionEvent
	done      bool
	err       error
	chunk     Chunk
}

type Chunk struct {
	Text string `json:"text"`
}

var completionSequence atomic.Uint64
var activeCompletions sync.Map

func lastError() error {
	message := C.GoString(C.llm_get_last_error_message())
	if message == "" {
		message = fmt.Sprintf("libllm error 0x%x", uint32(C.llm_get_last_error_code()))
	}
	return errors.New(message)
}

func cStringView(value string) (C.llm_string_view_t, unsafe.Pointer) {
	var view C.llm_string_view_t
	if value == "" {
		return view, nil
	}
	data := C.CBytes([]byte(value))
	view.data = (*C.char)(data)
	view.size = C.int64_t(len(value))
	return view, data
}

func parseDevice(device string) (C.llm_device_type_t, error) {
	switch strings.ToLower(device) {
	case "auto":
		return C.LLM_DEVICE_AUTO, nil
	case "cpu":
		return C.LLM_DEVICE_CPU, nil
	case "cuda":
		return C.LLM_DEVICE_CUDA, nil
	default:
		return C.LLM_DEVICE_AUTO, fmt.Errorf("invalid device: %s", device)
	}
}

func NewModel(filename, device string) (*Model, error) {
	if err := initLlm(); err != nil {
		return nil, err
	}
	deviceType, err := parseDevice(device)
	if err != nil {
		return nil, err
	}

	model := new(Model)
	if status := C.llm_engine_init(&model.engine); status != 0 {
		return nil, lastError()
	}

	var options C.llm_engine_options_t
	C.llm_engine_options_init(&options)
	modelPath, modelPathData := cStringView(filename)
	defer C.free(modelPathData)
	options.model_path = modelPath
	options.device = deviceType

	if status := C.llm_engine_load_go(&model.engine, &options); status != 0 {
		C.llm_engine_destroy(&model.engine)
		return nil, lastError()
	}

	runtime.SetFinalizer(model, func(model *Model) {
		_ = model.Close()
	})
	return model, nil
}

func (m *Model) Close() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	if m.closed {
		return nil
	}

	m.closed = true
	runtime.SetFinalizer(m, nil)
	status := C.llm_engine_destroy(&m.engine)
	if status != 0 {
		return lastError()
	}
	return nil
}

func DefaultCompletionConfig() CompletionConfig {
	return CompletionConfig{Temperature: 1.0, TopK: 0, TopP: 0.8}
}

func (m *Model) Complete(history []Message, config CompletionConfig) (*Completion, error) {
	if len(history) == 0 {
		return nil, errors.New("history is empty")
	}
	if config.Temperature < 0 || config.TopK < -1 || config.TopP <= 0 || config.TopP > 1 {
		return nil, errors.New("invalid completion config")
	}

	requestID := fmt.Sprintf("go-%d", completionSequence.Add(1))
	completion := &Completion{
		requestID: requestID,
	}
	completion.ready = sync.NewCond(&completion.mutex)
	activeCompletions.Store(requestID, completion)

	requestIDView, requestIDData := cStringView(requestID)
	defer C.free(requestIDData)

	messageBytes := C.size_t(len(history)) * C.size_t(unsafe.Sizeof(C.llm_message_t{}))
	messageData := C.malloc(messageBytes)
	if messageData == nil {
		activeCompletions.Delete(requestID)
		return nil, errors.New("unable to allocate message buffer")
	}
	defer C.free(messageData)
	messages := unsafe.Slice((*C.llm_message_t)(messageData), len(history))
	stringData := make([]unsafe.Pointer, 0, len(history)*2)
	defer func() {
		for _, data := range stringData {
			C.free(data)
		}
	}()
	for i, message := range history {
		role, roleData := cStringView(message.Role)
		content, contentData := cStringView(message.Content)
		messages[i].role = role
		messages[i].content = content
		if roleData != nil {
			stringData = append(stringData, roleData)
		}
		if contentData != nil {
			stringData = append(stringData, contentData)
		}
	}

	var request C.llm_request_t
	C.llm_request_init(&request)
	request.request_id = requestIDView
	request.messages = (*C.llm_message_t)(messageData)
	request.num_messages = C.int64_t(len(history))
	request.generation_config.temperature = C.float(config.Temperature)
	request.generation_config.top_k = C.int32_t(config.TopK)
	request.generation_config.top_p = C.float(config.TopP)

	m.mutex.Lock()
	defer m.mutex.Unlock()
	if m.closed {
		activeCompletions.Delete(requestID)
		return nil, errors.New("model is closed")
	}
	if status := C.llm_engine_add_request(&m.engine, &request); status != 0 {
		activeCompletions.Delete(requestID)
		return nil, lastError()
	}

	return completion, nil
}

func (c *Completion) Err() error {
	return c.err
}

func (c *Completion) Chunk() Chunk {
	return c.chunk
}

func (c *Completion) Next() bool {
	c.mutex.Lock()
	for len(c.events) == 0 && !c.done {
		c.ready.Wait()
	}
	if len(c.events) == 0 {
		c.mutex.Unlock()
		return false
	}
	event := c.events[0]
	c.events = c.events[1:]
	c.mutex.Unlock()

	if event.err != nil {
		c.err = event.err
	}
	if !event.hasChunk {
		return false
	}
	c.chunk = event.chunk
	return true
}
