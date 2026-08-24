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
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

var gInitMutex sync.Mutex
var gInitialized bool
var gDll unsafe.Pointer

// Initialize the libllm.
func initLlm() error {
	gInitMutex.Lock()
	defer gInitMutex.Unlock()
	if !gInitialized {
		// load the shared library.
		binPath, err := os.Executable()
		if err != nil {
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

		gDll = C.llm_load_library(dllPath)
		if gDll == nil {
			dllPath := C.CString("libllm.so")
			defer C.free(unsafe.Pointer(dllPath))

			gDll = C.llm_load_library(dllPath)
		}

		if gDll == nil {
			return errors.New("failed to load the libllm dynamic library")
		}

		// initialize the symbols.
		status := C.llm_load_symbols(gDll)
		if status != 0 {
			return errors.New("failed to load libllm api symbols")
		}

		// initialize libllm inference engine.
		C.llm_init()
		if C.llm_get_last_error_code() != 0 {
			return errors.New(C.GoString(C.llm_get_last_error_message()))
		}
		gInitialized = true
	}

	return nil
}
