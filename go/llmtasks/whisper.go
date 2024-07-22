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

package llmtasks

import (
	"bytes"
	"errors"
	"io"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/ling0322/libllm/go/llm"
)

var getFfmpegBin = sync.OnceValue[string](getFfmpegBinInternal)

var ErrInvalidWhisperSequence = errors.New("invalid sequence for Whisper model")
var ErrStreamIsNil = errors.New("input stream is nil")
var ErrWhisperModelIsNil = errors.New("whisper model is nil")

const (
	whisperStateAudio = iota
	whisperStateStartOfTranscription
	whisperStateLanguage
	whisperStateTranscribe
	whisperStateBeginTime
	whisperStateText
	whisperStateEndTime
)

// the transcriber with whisper model. implements the interface Transcriber.
type WhisperTranscriber struct {
	// the whisper model.
	WhisperModel llm.Model

	// the reader for input file.
	InputStream io.Reader

	// current state in whisper sequence decoding.
	state int

	// if any errors occured.
	err error

	// internal completion object for llm API.
	comp llm.Completion

	// the current transcription result.
	result TranscriptionResult

	// the predicted language.
	predictedLanguage string
}

// create a new instance of WhisperTranscriber from whisper model and stream of input file.
func NewWhisperTranscriber(whisperModel llm.Model, inputStream io.Reader) *WhisperTranscriber {
	return &WhisperTranscriber{
		WhisperModel: whisperModel,
		InputStream:  inputStream,
		state:        whisperStateAudio,
	}
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
func convertToPcm(inputStream io.Reader) ([]byte, error) {
	ffmpegBin := getFfmpegBin()
	if ffmpegBin == "" {
		return nil, errors.New("unable to find ffmpeg")
	}

	// ffmpeg found in the dir as llm, check it.
	cmd := exec.Command(
		ffmpegBin, "-hide_banner", "-nostdin", "-vn", "-threads", "0", "-i", "-", "-f",
		"s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-")
	cmd.Stdin = inputStream

	var dataBuffer, errBuffer bytes.Buffer
	cmd.Stdout = &dataBuffer
	cmd.Stderr = &errBuffer
	if err := cmd.Run(); err != nil {
		slog.Error("ffmpeg failed", "stderr", errBuffer.String())
		return nil, err
	}

	return dataBuffer.Bytes(), nil
}

// parse a whisper timestamp token like <|3.22|>. On success. return the (parsed-time, true). On
// failed, return (time.Duration(0), false)
func (w *WhisperTranscriber) parseTimestampToken(token string) (time.Duration, bool) {
	if token == "" {
		return time.Duration(0), false
	}

	if len(token) < 8 || token[:2] != "<|" || token[len(token)-2:] != "|>" {
		return time.Duration(0), false
	}

	offset, err := strconv.ParseFloat(token[2:len(token)-2], 64)
	if err != nil {
		return time.Duration(0), false
	}

	return time.Duration(math.Round(offset*1000) * float64(time.Millisecond)), true
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Transcribe() bool {
	if w.WhisperModel == nil {
		w.err = ErrWhisperModelIsNil
		return false
	}

	if w.InputStream == nil {
		w.err = ErrStreamIsNil
		return false
	}

	if w.state == whisperStateAudio {
		audio, err := convertToPcm(w.InputStream)
		if err != nil {
			w.err = err
			return false
		}

		audio = audio[:16000*30*2]

		prompt := llm.NewPrompt()
		prompt.AppendAudio(audio, llm.Pcm16kHz16BitMono)
		prompt.AppendControlToken("<|startoftranscript|>")

		compConfig := llm.NewCompletionConfig()
		compConfig.SetTopK(1)
		compConfig.SetTemperature(1.5)
		compConfig.SetConfig("generator.type", "whisper")
		comp, err := w.WhisperModel.Complete(compConfig, prompt)
		if err != nil {
			w.err = err
			return false
		}

		w.comp = comp
		w.state = whisperStateStartOfTranscription
	}

	result := TranscriptionResult{Language: w.predictedLanguage}
	for w.comp.Next() {
		token := w.comp.Token()
		piece := w.comp.Text()
		timeOffset, isTimestampToken := w.parseTimestampToken(token)
		if w.state == whisperStateStartOfTranscription && token == "<|nospeech|>" {
			// no speech, stop transcription.
			return false
		} else if w.state == whisperStateStartOfTranscription && len(token) == 6 {
			w.predictedLanguage = token[2 : len(token)-2]
			result.Language = w.predictedLanguage
			w.state = whisperStateLanguage
		} else if w.state == whisperStateStartOfTranscription {
			w.err = ErrInvalidWhisperSequence
			return false
		} else if w.state == whisperStateLanguage && token == "<|transcribe|>" {
			w.state = whisperStateTranscribe
		} else if w.state == whisperStateLanguage {
			w.err = ErrInvalidWhisperSequence
			return false
		} else if w.state == whisperStateTranscribe && isTimestampToken {
			result.Begin = timeOffset
			w.state = whisperStateBeginTime
		} else if w.state == whisperStateTranscribe {
			w.err = ErrInvalidWhisperSequence
			return false
		} else if w.state == whisperStateBeginTime && piece != "" {
			result.Text += piece
			w.state = whisperStateText
		} else if w.state == whisperStateBeginTime {
			w.err = ErrInvalidWhisperSequence
			return false
		} else if w.state == whisperStateText && piece != "" {
			result.Text += piece
		} else if w.state == whisperStateText && isTimestampToken {
			result.End = timeOffset
			w.state = whisperStateEndTime
			w.result = result

			return true
		} else if w.state == whisperStateText {
			// ignore other control tokens once transcription started.
		} else if w.state == whisperStateEndTime && isTimestampToken {
			result.Begin = timeOffset
			w.state = whisperStateBeginTime
		} else if w.state == whisperStateEndTime {
			w.err = ErrInvalidWhisperSequence
			return false
		} else {
			w.err = ErrInvalidWhisperSequence
			return false
		}
	}
	if err := w.comp.Error(); err != nil {
		w.err = err
	}

	if w.state != whisperStateEndTime {
		w.err = ErrInvalidWhisperSequence
	}

	return false
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Result() TranscriptionResult {
	return w.result
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Err() error {
	return w.err
}
