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
	"errors"

	"github.com/ling0322/libllm/go/llm"
)

type BilibiliIndex struct {
}

func (l *BilibiliIndex) Build(history []Message) (llm.Prompt, error) {
	prompt := llm.NewPrompt()
	if len(history) > 0 && history[0].Role == "system" {
		prompt.AppendControlToken("<unk>")
		prompt.AppendText(history[0].Content)
		history = history[1:]
	}

	for _, message := range history {
		if message.Role == "user" {
			prompt.AppendControlToken("<|reserved_0|>")
			prompt.AppendText(message.Content)
			prompt.AppendControlToken("<|reserved_1|>")
		} else if message.Role == "assistant" {
			prompt.AppendText(message.Content)
		} else {
			return nil, errors.New("unexpected role")
		}
	}

	return prompt, nil
}
