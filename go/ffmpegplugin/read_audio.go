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

package ffmpegplugin

// #cgo LDFLAGS: -ldl
// #cgo CXXFLAGS: -I/home/xiaoych/ffmpeg
// #include <string.h>
// #include <stdlib.h>
// #include "plugin.h"
import "C"
import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

var gInit atomic.Bool
var gDll unsafe.Pointer

var Init = sync.OnceValue[error](initIntrnal)

func initIntrnal() error {
	// load the shared library.
	binPath, err := os.Executable()
	if err != nil {
		return err
	}

	var libname string
	if runtime.GOOS == "windows" {
		libname = "llmpluginffmpeg.dll"
	} else if runtime.GOOS == "linux" {
		libname = "libllmpluginffmpeg.so"
	} else if runtime.GOOS == "darwin" {
		libname = "libllmpluginffmpeg.dylib"
	}

	binDir := filepath.Dir(binPath)
	dllPath := C.CString(filepath.Join(binDir, libname))
	defer C.free(unsafe.Pointer(dllPath))

	gDll = C.llm_ffmpeg_plugin_load_library(dllPath)
	if gDll == nil {
		return errors.New("failed to load the ffmpeg plugin")
	}

	// initialize the symbols.
	retcode := C.llm_ffmpeg_plugin_load_symbols(gDll)
	if retcode != 0 {
		return errors.New("failed to load ffmpeg plugin symbols")
	}

	gInit.Store(true)
	return nil
}

func Read16KHzMonoPcmFromMediaFile(filename string) (pcmdata []byte, err error) {
	if !gInit.Load() {
		return nil, errors.New("ffmpeg plugin not initialized")
	}

	cName := C.CString(filename)
	defer C.free(unsafe.Pointer(cName))

	var outputPtr *C.char
	outputLen := C.int(0)
	ret := C.llm_ffmpeg_plugin_read_16khz_mono_pcm_from_media_file(cName, &outputPtr, &outputLen)
	if ret < 0 {
		err = errors.New(C.GoString(C.llm_ffmpeg_plugin_get_err()))
		return
	}

	pcmdata = make([]byte, int(outputLen))
	C.memcpy(unsafe.Pointer(&pcmdata[0]), unsafe.Pointer(outputPtr), C.size_t(outputLen))
	return
}
