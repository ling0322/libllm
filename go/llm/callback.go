// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

// #include "llm.h"
import "C"

import (
	"errors"
	"unsafe"
)

func goString(view C.llm_string_view_t) string {
	if view.data == nil || view.size == 0 {
		return ""
	}
	return C.GoStringN(view.data, C.int(view.size))
}

//export goLlmStreamCallback
func goLlmStreamCallback(outputs *C.llm_request_outputs_t, _ unsafe.Pointer) {
	if outputs == nil || outputs.data == nil || outputs.size <= 0 {
		return
	}

	batch := unsafe.Slice(outputs.data, int(outputs.size))
	for i := range batch {
		output := &batch[i]
		requestID := goString(output.request_id)
		value, ok := activeCompletions.Load(requestID)
		if !ok {
			continue
		}
		completion := value.(*Completion)

		event := completionEvent{}
		if output.num_token_ids > 0 {
			event.chunk.Text = goString(output.text)
			event.hasChunk = true
		}
		if output.error_message.size > 0 {
			event.err = errors.New(goString(output.error_message))
		}
		completion.mutex.Lock()
		completion.events = append(completion.events, event)
		if output.finished != 0 {
			completion.done = true
			activeCompletions.Delete(requestID)
		}
		completion.ready.Broadcast()
		completion.mutex.Unlock()
	}
}
