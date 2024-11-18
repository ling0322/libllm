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
// const int cLLM_ERROR_EOF = LLM_ERROR_EOF;
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"time"
	"unsafe"
)

// A LLM.
type ASRModel struct {
	model C.llm_asr_model_t
}

type RecognitionResult struct {
	Text     string
	Language string
	Begin    time.Duration
	End      time.Duration
}

type llmRecognitionResultJson struct {
	Text     string `json:"text"`
	Language string `json:"language"`
	BeginMs  int64  `json:"begin"`
	EndMs    int64  `json:"end"`
}

type Recognition struct {
	recognition C.llm_asr_recognizer_t
	err         error
	result      RecognitionResult
	json        *llmJson
}

type llmJson struct {
	json C.llm_json_t
}

func newRecognition() *Recognition {
	r := &Recognition{}
	r.json = newJson()
	C.llm_asr_recognizer_init(&r.recognition)
	runtime.SetFinalizer(r, func(r *Recognition) {
		C.llm_asr_recognizer_destroy(&r.recognition)
	})

	return r
}

func newJson() *llmJson {
	j := &llmJson{}
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
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return nil
}

func (j *llmJson) unmarshal(v any) error {
	bufSize := 2048
	buf := C.malloc(C.size_t(bufSize))
	defer C.free(buf)

	status := C.llm_json_dump(&j.json, (*C.char)(buf), C.int64_t(bufSize))
	if status != 0 {
		return errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	jsonStr := C.GoString((*C.char)(buf))
	return json.Unmarshal([]byte(jsonStr), v)
}

func NewASRModel(filename, device string) (*ASRModel, error) {
	err := initLlm()
	if err != nil {
		return nil, err
	}

	json := newJson()
	json.marshal(map[string]any{
		"filename": filename,
		"device":   device,
	})

	m := &ASRModel{}
	C.llm_asr_model_init(&m.model)
	runtime.SetFinalizer(m, func(m *ASRModel) {
		C.llm_asr_model_destroy(&m.model)
	})

	status := C.llm_asr_model_load(&m.model, &json.json)
	if status != 0 {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return m, nil
}

func (m *ASRModel) Recognize(filename string) (*Recognition, error) {
	r := newRecognition()

	json := newJson()
	json.marshal(map[string]any{
		"media_file": filename,
	})

	status := C.llm_asr_recognize_media_file(&r.recognition, &m.model, &json.json)
	if status != 0 {
		return nil, errors.New(C.GoString(C.llmGetLastErrorMessage()))
	}

	return r, nil
}

func (r *Recognition) Err() error {
	return r.err
}

func (r *Recognition) Result() *RecognitionResult {
	return &r.result
}

func (r *Recognition) Next() bool {
	status := C.llm_asr_recognizer_get_next_result(&r.recognition, &r.json.json)
	if status == C.cLLM_ERROR_EOF {
		return false
	} else if status != 0 {
		r.err = errors.New(C.GoString(C.llmGetLastErrorMessage()))
		return false
	}

	rawReco := &llmRecognitionResultJson{}
	r.err = r.json.unmarshal(&rawReco)
	if r.err != nil {
		return false
	}

	r.result.Text = rawReco.Text
	r.result.Language = rawReco.Language
	r.result.Begin = time.Duration(rawReco.BeginMs * int64(time.Millisecond))
	r.result.End = time.Duration(rawReco.EndMs * int64(time.Millisecond))

	return true
}
