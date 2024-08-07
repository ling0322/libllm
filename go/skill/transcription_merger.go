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
	"log"
	"strings"
	"time"
	"unicode"
)

type TranscriptionMergeOptions struct {
	Language Lang
}

type TranscriptionMerger struct {
	options TranscriptionMergeOptions
}

func NewTranscriptionMerger(options TranscriptionMergeOptions) *TranscriptionMerger {
	return &TranscriptionMerger{
		options: options,
	}
}

func (m *TranscriptionMerger) isBoundaryPunctuationEn(ch rune) bool {
	switch ch {
	case '.':
		return true
	case '!':
		return true
	case '?':
		return true
	case '"':
		return true
	default:
		return false
	}
}

func (m *TranscriptionMerger) isSoftBoundaryPunctuationEn(ch rune) bool {
	switch ch {
	case ',':
		return true
	case ';':
		return true
	case ':':
		return true
	default:
		return false
	}
}

func (m *TranscriptionMerger) maybeSentenceBoundaryEn(left, right TranscriptionResult) bool {
	dt := right.Begin - left.End
	if dt > time.Second*3 {
		return true
	}

	flagLongSil := false
	if dt > time.Second*2 {
		flagLongSil = true
	}

	leftText := []rune(strings.TrimSpace(left.Text))
	rightText := []rune(strings.TrimSpace(right.Text))
	flagBoundaryPunctL := m.isBoundaryPunctuationEn(leftText[len(leftText)-1])
	flagSoftBoundaryPunctL := m.isSoftBoundaryPunctuationEn(leftText[len(leftText)-1])
	flagCapitalR := unicode.IsUpper(rightText[0])
	flagLetterR := unicode.IsLetter(rightText[0])
	flagLongTranscriptionL := len(strings.Split(left.Text, " ")) > 30
	flagLongAudioLengthL := left.Duration() > 15*time.Second

	switch {
	case flagLongSil && flagBoundaryPunctL:
		return true
	case flagLongSil && flagCapitalR:
		return true
	case flagLongSil && !flagLetterR:
		return true
	case flagBoundaryPunctL && flagCapitalR:
		return true
	case flagBoundaryPunctL && !flagLetterR:
		return true
	case flagLongTranscriptionL && flagSoftBoundaryPunctL:
		return true
	case flagLongTranscriptionL && flagBoundaryPunctL:
		return true
	case flagLongAudioLengthL:
		return true
	default:
		return true
	}
}

func (m *TranscriptionMerger) maybeSentenceBoundary(left, right TranscriptionResult) bool {
	switch m.options.Language {
	case English:
		return m.maybeSentenceBoundaryEn(left, right)
	default:
		log.Fatal("unsupported language for sentence boundary detection")
	}

	return false
}

func (m *TranscriptionMerger) mergePair(left, right TranscriptionResult) TranscriptionResult {
	merged := TranscriptionResult{}
	merged.Begin = left.Begin
	merged.End = right.End

	merged.Text = left.Text
	switch m.options.Language {
	case English:
		merged.Text += " "
	}

	merged.Text += right.Text
	merged.Language = left.Language

	return merged
}

func (m *TranscriptionMerger) Merge(transcriptions []TranscriptionResult) []TranscriptionResult {
	mergedTranscriptions := []TranscriptionResult{}
	for _, t := range transcriptions {
		if len(mergedTranscriptions) == 0 {
			mergedTranscriptions = append(mergedTranscriptions, t)
			continue
		}

		left := mergedTranscriptions[len(mergedTranscriptions)-1]
		switch {
		case m.maybeSentenceBoundary(left, t):
			mergedTranscriptions = append(mergedTranscriptions, t)
		default:
			mergedTranscriptions[len(mergedTranscriptions)-1] = m.mergePair(left, t)
		}
	}

	return mergedTranscriptions
}
