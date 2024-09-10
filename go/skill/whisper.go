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
	"compress/zlib"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/ling0322/libllm/go/llm"
)

var regexpLangToken = regexp.MustCompile(`^<\|([a-z][a-z][a-z]?)\|>$`)
var ErrInvalidWhisperSequence = errors.New("invalid sequence for Whisper model")
var ErrStreamIsEmpty = errors.New("input stream is nil")
var ErrWhisperModelIsNil = errors.New("whisper model is nil")
var ErrNoMoreResults = errors.New("no more results")
var ErrAudioEndOfStream = errors.New("audio end of stream")

const SampleRate = 16000
const BytesPerSample = 2

// the transcriber with whisper model. implements the interface Transcriber.
type WhisperTranscriber struct {
	// the whisper model.
	WhisperModel llm.Model

	// the reader for input file.
	stream *WaveStream

	streamOffset time.Duration

	chunk WaveChunk

	// if any errors occured.
	err error

	// internal completion object for llm API.
	comp llm.Completion

	// the current transcription result.
	result TranscriptionResult

	// the predicted language.
	predictedLanguage string

	// the specified whisper language. Empty means let whisper to predict.
	language string

	// the transcription history.
	history []string
}

// create a new instance of WhisperTranscriber from whisper model and stream of input file.
func NewWhisperTranscriber(whisperModel llm.Model, inputFile string) (*WhisperTranscriber, error) {
	stream, err := NewWaveStream(inputFile)
	if err != nil {
		return nil, err
	}

	return &WhisperTranscriber{
		WhisperModel: whisperModel,
		stream:       stream,
	}, nil
}

func (w *WhisperTranscriber) SetLanguage(langCode string) {
	w.language = langCode
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
	result.Begin = w.stream.Offset() + beginOffset
	if w.chunk.Duration()-beginOffset < 200*time.Millisecond {
		slog.Info("return ErrNoMoreResults", "begin", result.Begin, "audio_len", w.chunk.end)
		return TranscriptionResult{}, ErrNoMoreResults
	}

	transcriptionDone := false
	for w.comp.Next() {
		token := w.comp.Token()
		piece := w.comp.Text()
		slog.Debug("comp.next()", "token", token, "piece", piece)
		offset, isTimestampToken := w.parseTimestampToken(token)
		if isTimestampToken {
			result.End = w.stream.Offset() + offset
			transcriptionDone = true
			break
		}

		result.Text += piece
	}
	if w.comp.Error() != nil {
		return TranscriptionResult{}, err
	} else if !transcriptionDone {
		// generation stops at half of the transcription.
		return TranscriptionResult{}, ErrNoMoreResults
	}

	result.End = min(result.End, w.chunk.end)
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
	if w.chunk.eof {
		// if the current chunk is already the last one.
		return ErrAudioEndOfStream
	}

	err := w.stream.Seek(w.streamOffset)
	if errors.Is(err, io.EOF) {
		return ErrAudioEndOfStream
	} else if err != nil {
		return err
	}

	slog.Info("transcribe segment", "offset", w.stream.Offset())
	w.chunk, err = w.stream.ReadChunk(30 * time.Second)
	if errors.Is(err, io.EOF) {
		return ErrAudioEndOfStream
	} else if w.chunk.Duration() < 500*time.Millisecond {
		slog.Info("w.chunk.Duration() < 500*time.Millisecond")
		return ErrAudioEndOfStream
	} else if err != nil {
		return err
	}

	prompt := llm.NewPrompt()
	prompt.AppendAudio(w.chunk.data, llm.Pcm16kHz16BitMono)
	prompt.AppendControlToken("<|startoftranscript|>")

	compConfig := llm.NewCompletionConfig()
	compConfig.SetTopK(1)
	compConfig.SetConfig("generator.type", "whisper")
	if w.language != "" {
		compConfig.SetConfig("whisper.language", w.language)
	}
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
	if w.err != nil {
		return false
	}

	if w.WhisperModel == nil {
		w.err = ErrWhisperModelIsNil
		return false
	}

	if w.stream == nil {
		w.err = ErrStreamIsEmpty
		return false
	}

	// loop until we got a valid transcription.
	for {
		beginOfSegment := false
		if w.comp == nil {
			w.err = w.prefillNextAudioSegment()
			if errors.Is(w.err, ErrNoMoreResults) {
				slog.Info("segment end", "reason", "ErrNoMoreResultsPrefill")
				w.disposeCompAndSetToNil()
				w.streamOffset += 30 * time.Second
				continue
			} else if errors.Is(w.err, ErrAudioEndOfStream) {
				slog.Info("segment end", "reason", "ErrAudioEndOfStream")
				w.disposeCompAndSetToNil()
				w.err = nil
				return false
			} else if w.err != nil {
				return false
			}
			beginOfSegment = true
			w.resetRepetitionChecker()
		}

		result, err := w.decodeTranscription()
		if errors.Is(err, ErrNoMoreResults) && beginOfSegment {
			// if no result for the whole audio segment, move forward to the next 30s segment.
			slog.Info("segment end", "reason", "ErrNoMoreResults")
			w.disposeCompAndSetToNil()
			w.streamOffset += 30 * time.Second
			continue
		} else if errors.Is(err, ErrNoMoreResults) && !beginOfSegment {
			// move the wave offset to the end of last completed transcription.
			slog.Info("segment end", "reason", "ErrNoMoreResults")
			w.disposeCompAndSetToNil()
			w.streamOffset = w.result.End
			continue
		} else if err != nil {
			w.err = err
			return false
		}

		if w.checkRepetition(result) {
			// once repetition happened, stop the current decoding process.
			slog.Info("segment end", "reason", "RepetitionDetected")
			w.disposeCompAndSetToNil()
			w.streamOffset = w.result.End
			continue
		}

		w.result = result
		return true
	}
}

func (w *WhisperTranscriber) resetRepetitionChecker() {
	w.history = []string{}
}

func (w *WhisperTranscriber) checkRepetition(r TranscriptionResult) bool {
	w.history = append(w.history, r.Text)
	if len(w.history) > 200 {
		w.history = w.history[1:]
	}

	text := strings.Join(w.history, " ")

	// compress text
	var b bytes.Buffer

	writer := zlib.NewWriter(&b)
	writer.Write([]byte(text))
	writer.Close()

	compressionRatio := float32(len(text)) * 1.0 / float32(len(b.Bytes()))
	if compressionRatio > 2.4 {
		slog.Info("check repetition", "compression_ratio", compressionRatio)
		return true
	} else {
		return false
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
	return w.stream.Offset()
}
