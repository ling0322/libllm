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
	"fmt"

	"github.com/ling0322/libllm/go/llm"
)

type Qwen struct {
}

func (l *Qwen) Build(history []Message) (llm.Prompt, error) {
	if len(history) == 0 {
		return nil, errors.New("history is empty")
	}

	prompt := llm.NewPrompt()
	for _, message := range history[:len(history)-1] {
		prompt.AppendControlToken("<|im_start|>")
		prompt.AppendText(fmt.Sprintf("%s\n%s", message.Role, message.Content))
		prompt.AppendControlToken("<|im_end|>")
		prompt.AppendText("\n")
	}

	lastMessage := history[len(history)-1]
	if lastMessage.Role == "user" {
		prompt.AppendControlToken("<|im_start|>")
		prompt.AppendText(fmt.Sprintf("%s\n%s", lastMessage.Role, lastMessage.Content))
		prompt.AppendControlToken("<|im_end|>")
		prompt.AppendText("\n")
		prompt.AppendControlToken("<|im_start|>")
		prompt.AppendText("assistant\n")
	} else if lastMessage.Role == "assistant" {
		prompt.AppendControlToken("<|im_start|>")
		prompt.AppendText(fmt.Sprintf("%s\n%s", lastMessage.Role, lastMessage.Content))
	} else {
		return nil, errors.New("last message should be either user or assistant")
	}

	return prompt, nil
}
