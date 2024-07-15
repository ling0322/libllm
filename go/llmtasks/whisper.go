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
	"fmt"

	"github.com/ling0322/libllm/go/llm"
)

type Whisper struct {
	model llm.Model
}

func NewWhisper(model llm.Model) Transcriber {
	return &Whisper{model}
}

func (w *Whisper) languageToTag(language string) (string, error) {
	if language == "english" {
		return "<|en|>", nil
	}
	if language == "chinese" {
		return "<|zh|>", nil
	}

	return "", fmt.Errorf("unexpected language: %s", language)
}

func (w *Whisper) Transcribe(audio []byte, config TranscriptionConfig) (llm.Completion, error) {
	prompt := llm.NewPrompt()
	prompt.AppendAudio(audio, llm.Pcm16kHz16BitMono)
	if config.Prompt != "" {
		prompt.AppendControlToken("<|prev|>")
		prompt.AppendText(config.Prompt)
	}

	prompt.AppendControlToken("<|startoftranscript|>")
	// languageTag, err := w.languageToTag(config.Language)
	// if err != nil {
	// 	return nil, err
	//}
	// prompt.AppendControlToken(languageTag)
	// prompt.AppendControlToken("<|transcribe|>")

	compConfig := llm.NewCompletionConfig()
	compConfig.SetTopK(1)
	compConfig.SetTemperature(1.5)
	comp, err := w.model.Complete(compConfig, prompt)
	return comp, err
}
