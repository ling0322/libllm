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
// #include "llm.h"
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

type llmJson struct {
	json C.llm_json_t
}

func newJson() *llmJson {
	j := new(llmJson)
	C.llm_json_init(&j.json)
	runtime.SetFinalizer(j, func(j *llmJson) {
		C.llm_json_destroy(&j.json)
	})

	return j
}

func (j *llmJson) marshal(v any) error {
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return err
	}

	cJsonStr := C.CString(string(jsonBytes))
	defer C.free(unsafe.Pointer(cJsonStr))

	status := C.llm_json_parse(&j.json, cJsonStr)
	if status != 0 {
		return errors.New(C.GoString(C.llm_get_last_error_message()))
	}

	return nil
}

func (j *llmJson) unmarshal(v any) error {
	bufSize := 2048
	buf := C.malloc(C.size_t(bufSize))
	defer C.free(buf)

	status := C.llm_json_dump(&j.json, (*C.char)(buf), C.int64_t(bufSize))
	if status != 0 {
		return errors.New(C.GoString(C.llm_get_last_error_message()))
	}

	jsonStr := C.GoString((*C.char)(buf))
	return json.Unmarshal([]byte(jsonStr), v)
}
