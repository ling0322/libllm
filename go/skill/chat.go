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
	"github.com/ling0322/libllm/go/llm"
)

type Message struct {
	Role    string
	Content string
}

type Chat struct {
	model         llm.Model
	promptBuilder promptBuilder
	compConfig    llm.CompletionConfig
}

func NewChat(model llm.Model) (*Chat, error) {
	modelName := model.GetName()
	promptBuilder, err := newPromptBuilder(modelName)
	if err != nil {
		return nil, err
	}

	return &Chat{
		model:         model,
		promptBuilder: promptBuilder,
		compConfig:    llm.NewCompletionConfig(),
	}, nil
}

func (c *Chat) Chat(history []Message) (llm.Completion, error) {
	prompt, err := c.promptBuilder.Build(history)
	if err != nil {
		return nil, err
	}

	comp, err := c.model.Complete(c.compConfig, prompt)
	if err != nil {
		return nil, err
	}

	return comp, nil
}
