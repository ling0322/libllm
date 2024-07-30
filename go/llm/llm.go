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
// #include "llm_api.h"
import "C"
import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"unsafe"
)

type Device int32
type AudioFormat int32

const (
	Cpu               = Device(0x0000)
	Cuda              = Device(0x0100)
	Auto              = Device(0x1f00)
	Pcm16kHz16BitMono = AudioFormat(0x0001)
)

var gInit atomic.Bool
var gDll unsafe.Pointer

// Initialize the libllm.
func initLlm() error {
	if !gInit.Swap(true) {
		// load the shared library.
		binPath, err := os.Executable()
		if err != nil {
			gInit.Store(false)
			return err
		}

		var libname string
		if runtime.GOOS == "windows" {
			libname = "llm.dll"
		} else if runtime.GOOS == "linux" {
			libname = "libllm.so"
		} else if runtime.GOOS == "darwin" {
			libname = "libllm.dylib"
		}

		binDir := filepath.Dir(binPath)
		dllPath := C.CString(filepath.Join(binDir, libname))
		defer C.free(unsafe.Pointer(dllPath))

		gDll = C.llmLoadLibrary(dllPath)
		if gDll == nil {
			dllPath := C.CString("libllm.so")
			defer C.free(unsafe.Pointer(dllPath))

			gDll = C.llmLoadLibrary(dllPath)
		}

		if gDll == nil {
			gInit.Store(false)
			return errors.New("failed to load the libllm dynamic library")
		}

		// initialize the symbols.
		status := C.llmLoadSymbols(gDll)
		if status != C.LLM_OK {
			gInit.Store(false)
			return errors.New("failed to load libllm api symbols")
		}

		// initialize libllm inference engine.
		status = C.llmInit(C.LLM_API_VERSION)
		if status != C.LLM_OK {
			gInit.Store(false)
			return errors.New(C.GoString(C.llmGetLastErrorMessage()))
		}
	}

	return nil
}

// Release all the resources allocated in libllm library.
func Release() {
	if gInit.Swap(false) {
		status := C.llmDestroy()
		if status != C.LLM_OK {
			fmt.Fprintf(
				os.Stderr,
				"failed to destroy libllm: %s\n",
				C.GoString(C.llmGetLastErrorMessage()))
		}

		// release the dynamic library itself.
		status = C.llmDestroyLibrary(gDll)
		if status != C.LLM_OK {
			fmt.Fprintf(os.Stderr, "failed to close dynamic library of libllm\n")
		}
	}
}
