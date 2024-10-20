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

type Llama struct {
}

func (l *Llama) Build(history []Message) (llm.Prompt, error) {
	prompt := llm.NewPrompt()
	prompt.AppendControlToken("<|begin_of_text|>")
	for _, message := range history[:len(history)-1] {
		prompt.AppendControlToken("<|start_header_id|>")
		prompt.AppendText(message.Role)
		prompt.AppendControlToken("<|end_header_id|>")
		prompt.AppendText("\n\n" + message.Content)
		prompt.AppendControlToken("<|eot_id|>")
	}

	lastMessage := history[len(history)-1]
	if lastMessage.Role == "user" {
		prompt.AppendControlToken("<|start_header_id|>")
		prompt.AppendText(lastMessage.Role)
		prompt.AppendControlToken("<|end_header_id|>")
		prompt.AppendText("\n\n" + lastMessage.Content)
		prompt.AppendControlToken("<|eot_id|>")
		prompt.AppendControlToken("<|start_header_id|>")
		prompt.AppendText("assistant")
		prompt.AppendControlToken("<|end_header_id|>")
		prompt.AppendText("\n\n")
	} else if lastMessage.Role == "assistant" {
		prompt.AppendControlToken("<|start_header_id|>")
		prompt.AppendText(lastMessage.Role)
		prompt.AppendControlToken("<|end_header_id|>")
		prompt.AppendText("\n\n" + lastMessage.Content)
	} else {
		return nil, errors.New("last message should be either user or assistant")
	}

	return prompt, nil
}
