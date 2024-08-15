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

package skill

import (
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ling0322/libllm/go/ffmpegplugin"
)

var gFfmpegBin string
var gFfmpegPluginReady bool

var initAudioReader = sync.OnceFunc(func() {
	err := ffmpegplugin.Init()
	if err != nil {
		slog.Warn(fmt.Sprintf("load ffmpeg plugin failed: %s", err))
	} else {
		gFfmpegPluginReady = true
	}

	gFfmpegBin = getFfmpegBinInternal()
	if gFfmpegBin == "" {
		slog.Warn("unable to find ffmpeg")
	}
})

// convert the input file to pcm .wav file in OS temporary directory using ffmpeg.
func convertToPcmPlugin(inputFile string) ([]byte, error) {
	return ffmpegplugin.Read16KHzMonoPcmFromMediaFile(inputFile)
}

// find the path of ffmpeg.
func getFfmpegBinInternal() string {
	ffmpegBin := "ffmpeg"
	if runtime.GOOS == "windows" {
		ffmpegBin += ".exe"
	}

	cmd := exec.Command(ffmpegBin, "-version")
	err := cmd.Run()
	if err == nil {
		// ffmpeg in $PATH
		return ffmpegBin
	}

	binPath, err := os.Executable()
	if err != nil {
		return ""
	}

	binDir := filepath.Dir(binPath)
	ffmpegPath := filepath.Join(binDir, ffmpegBin)
	_, err = os.Stat(ffmpegPath)
	if err != nil {
		return ""
	}

	// ffmpeg found in the dir as llm, check it.
	cmd = exec.Command(ffmpegPath, "-version")
	err = cmd.Run()
	if err != nil {
		return ""
	}

	return ffmpegPath
}

// convert the input file to pcm .wav file in OS temporary directory using ffmpeg.
func convertToPcmBin(inputFile string) ([]byte, error) {
	if gFfmpegBin == "" {
		return nil, errors.New("unable to find ffmpeg")
	}

	// ffmpeg found in the dir as llm, check it.
	cmd := exec.Command(
		gFfmpegBin, "-hide_banner", "-nostdin", "-vn", "-threads", "0", "-i", inputFile, "-f",
		"s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-")
	var dataBuffer, errBuffer bytes.Buffer
	cmd.Stdout = &dataBuffer
	cmd.Stderr = &errBuffer
	if err := cmd.Run(); err != nil {
		slog.Error("ffmpeg failed", "stderr", errBuffer.String())
		return nil, err
	}

	return dataBuffer.Bytes(), nil
}

// read audio from media file and return as bytes of 16KHz, 16bit, mono-channel PCM.
func ReadAudioFromMediaFile(inputFile string) ([]byte, error) {
	initAudioReader()

	if gFfmpegPluginReady {
		return convertToPcmPlugin(inputFile)
	}

	if gFfmpegBin != "" {
		return convertToPcmBin(inputFile)
	}

	return nil, errors.New("unable to read media file since neither ffmpeg binary nor plugin found")
}
