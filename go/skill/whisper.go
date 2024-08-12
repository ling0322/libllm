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
	"io"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/ling0322/libllm/go/llm"
)

var regexpLangToken = regexp.MustCompile(`^<\|([a-z][a-z][a-z]?)\|>$`)
var getFfmpegBin = sync.OnceValue[string](getFfmpegBinInternal)

var ErrInvalidWhisperSequence = errors.New("invalid sequence for Whisper model")
var ErrStreamIsNil = errors.New("input stream is nil")
var ErrWhisperModelIsNil = errors.New("whisper model is nil")
var ErrNoMoreResults = errors.New("no more results")
var ErrAudioEndOfStream = errors.New("audio end of stream")

const SampleRate = 16000
const BytesPerSample = 2

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

	// the wave bytes for decoding. The format is 16khz 16bit mono-channel PCM without headfer.
	wavePayload []byte

	// offset of the current segment in wavePayload.
	waveOffset time.Duration
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

// complete next token from whisper model. If completion ends, return ErrNoMoreResults. For other
// errors, return it directly.
func (w *WhisperTranscriber) completeNext() error {
	ok := w.comp.Next()
	slog.Debug("completeNext()", "token", w.comp.Token(), "piece", w.comp.Text())
	if w.comp.Error() != nil {
		return w.comp.Error()
	} else if !ok {
		return ErrNoMoreResults
	}

	return nil
}

// decode one transcription from Whisper. Here transcription means a piece of text wrapped by begin
// and end timestamp tokens. On success, returns the result. If the whisper model generation ends
// in the begining or half of the transcription, return ErrNoMoreResults.
func (w *WhisperTranscriber) decodeTranscription() (TranscriptionResult, error) {
	result := TranscriptionResult{Language: w.predictedLanguage}

	err := w.completeNext()
	if err != nil {
		return TranscriptionResult{}, err
	}

	beginOffset, ok := w.parseTimestampToken(w.comp.Token())
	if !ok {
		return TranscriptionResult{}, fmt.Errorf("%w: not a time token", ErrInvalidWhisperSequence)
	}
	result.Begin = w.waveOffset + beginOffset

	transcriptionDone := false
	for w.comp.Next() {
		token := w.comp.Token()
		piece := w.comp.Text()
		slog.Debug("comp.next()", "token", token, "piece", piece)
		offset, isTimestampToken := w.parseTimestampToken(token)
		if isTimestampToken {
			result.End = w.waveOffset + offset
			transcriptionDone = true
			break
		}

		result.Text += piece
	}
	slog.Info("<EOT> got")
	if w.comp.Error() != nil {
		return TranscriptionResult{}, err
	} else if !transcriptionDone {
		// generation stops at half of the transcription.
		return TranscriptionResult{}, ErrNoMoreResults
	}

	return result, nil
}

// parse a language token and return the language name.
func (w *WhisperTranscriber) parseLanguageToken(token string) (lang string, ok bool) {
	match := regexpLangToken.FindStringSubmatch(token)
	if len(match) == 0 {
		return "", false
	}

	return match[1], true
}

// prefill audio and prompt when in the begining of decoding or last audio segment finished. If no
// transcriotion result or <|nospeech|> got, return ErrNoMoreResults.
func (w *WhisperTranscriber) prefillNextAudioSegment() error {
	nsPerSample := 1000000000 / SampleRate
	sampleOffset := int(w.waveOffset.Nanoseconds() / int64(nsPerSample))
	byteOffset := sampleOffset * 2
	if len(w.wavePayload)-byteOffset < SampleRate/10 {
		// ignore the last segment that less than 0.1s.
		return ErrAudioEndOfStream
	}
	slog.Info("prefill segment", "offset", w.waveOffset, "byteOffset", byteOffset)

	nBytes := min(len(w.wavePayload)-byteOffset, 30*SampleRate*2)
	audio := w.wavePayload[byteOffset : byteOffset+nBytes]

	prompt := llm.NewPrompt()
	prompt.AppendAudio(audio, llm.Pcm16kHz16BitMono)
	prompt.AppendControlToken("<|startoftranscript|>")

	compConfig := llm.NewCompletionConfig()
	compConfig.SetTopK(1)
	compConfig.SetConfig("generator.type", "whisper")
	comp, err := w.WhisperModel.Complete(compConfig, prompt)
	if err != nil {
		return err
	}

	// first token would be language token or <|nospeech|>
	w.comp = comp
	err = w.completeNext()
	if err != nil {
		return err
	}

	// exit once no speech for the whole audio segment.
	if w.comp.Token() == "<|nospeech|>" {
		return ErrNoMoreResults
	}

	// language token.
	lang, ok := w.parseLanguageToken(w.comp.Token())
	if !ok {
		return fmt.Errorf("%w: not a language token", ErrInvalidWhisperSequence)
	}
	w.predictedLanguage = lang

	// transcribe token
	err = w.completeNext()
	if err != nil {
		return err
	}

	if w.comp.Token() != "<|transcribe|>" {
		return fmt.Errorf("%w: not a transcribe token", ErrInvalidWhisperSequence)
	}

	// setup the compeltion done.
	return nil
}

func (w *WhisperTranscriber) disposeCompAndSetToNil() {
	if w.comp != nil {
		w.comp.Dispose()
		w.comp = nil
	}
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Transcribe() bool {
	if w.wavePayload == nil {
		w.wavePayload, w.err = convertToPcm(w.InputStream)
	}
	if w.err != nil {
		return false
	}

	if w.WhisperModel == nil {
		w.err = ErrWhisperModelIsNil
		return false
	}

	if w.InputStream == nil {
		w.err = ErrStreamIsNil
		return false
	}

	// loop until we got a valid transcription.
	for {
		beginOfSegment := false
		if w.comp == nil {
			w.err = w.prefillNextAudioSegment()
			beginOfSegment = true
		}

		if errors.Is(w.err, ErrNoMoreResults) {
			w.disposeCompAndSetToNil()
			w.err = nil
			w.waveOffset += 30 * time.Second
			continue
		} else if errors.Is(w.err, ErrAudioEndOfStream) {
			w.disposeCompAndSetToNil()
			w.err = nil
			return false
		} else if w.err != nil {
			return false
		}

		result, err := w.decodeTranscription()
		if errors.Is(err, ErrNoMoreResults) && beginOfSegment {
			// if no result for the whole audio segment, move forward to the next 30s segment.
			w.disposeCompAndSetToNil()
			w.waveOffset += 30 * time.Second
			continue
		} else if errors.Is(err, ErrNoMoreResults) && !beginOfSegment {
			// move the wave offset to the end of last completed transcription.
			w.disposeCompAndSetToNil()
			w.waveOffset = w.result.End
			continue
		} else if err != nil {
			w.err = err
			return false
		}

		w.result = result
		return true
	}
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Result() TranscriptionResult {
	return w.result
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Err() error {
	return w.err
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Dispose() {
	if w.comp != nil {
		w.disposeCompAndSetToNil()
	}
}

// implements interface Transcriber.
func (w *WhisperTranscriber) Offset() time.Duration {
	return w.waveOffset
}
