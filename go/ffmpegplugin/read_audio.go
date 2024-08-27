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

// #cgo linux LDFLAGS: -ldl
// #include <string.h>
// #include <stdlib.h>
// #include "plugin.h"
import "C"
import (
	"errors"
	"io"
	"log/slog"
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

type Reader struct {
	handle unsafe.Pointer
}

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

func NewReader(filename string) (*Reader, error) {
	Init()

	if !gInit.Load() {
		return nil, errors.New("ffmpeg plugin not initialized")
	}

	cName := C.CString(filename)
	defer C.free(unsafe.Pointer(cName))

	handle := C.llm_ffmpeg_audio_open(cName)
	if handle == nil {
		return nil, errors.New(C.GoString(C.llm_ffmpeg_get_err()))
	}

	reader := &Reader{
		unsafe.Pointer(handle),
	}
	runtime.SetFinalizer(reader, func(r *Reader) {
		if r.handle != nil {
			slog.Warn("ffmpegplugin.Reader is not closed")
			r.Close()
		}
	})
	return reader, nil
}

func (r *Reader) Read(b []byte) (n int, err error) {
	if r.handle == nil {
		return 0, errors.New("llm_ffmpeg_audio_reader_t handle is empty")
	}

	buf := (*C.char)(unsafe.Pointer(&b[0]))
	bufsize := C.int32_t(len(b))

	nb := C.llm_ffmpeg_audio_read(r.handle, buf, bufsize)
	if nb == 0 {
		return 0, io.EOF
	} else if nb < 0 {
		return 0, errors.New(C.GoString(C.llm_ffmpeg_get_err()))
	} else {
		return int(nb), nil
	}
}

func (r *Reader) Close() error {
	if r.handle != nil {
		C.llm_ffmpeg_audio_close(r.handle)
		r.handle = nil
	}

	return nil
}
