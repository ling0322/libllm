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
	"fmt"
	"math/rand"

	"github.com/ling0322/libllm/go/llm"
)

type TranslationRequest struct {
	Text              string
	LeftContextSource string
	LeftContextTarget string
	SourceLang        Lang
	TargetLang        Lang
	Temperature       float32
}

type Translator interface {
	Translate(request TranslationRequest) (llm.Completion, error)

	// return true if the translator supports the source and target language pairs.
	IsSupport(source, target Lang) bool
}

// translator implemented by a chat model
type chatTranslator struct {
	model llm.Model
}

func NewTranslator(model llm.Model) (Translator, error) {
	if model == nil {
		return nil, ErrModelIsNil
	}

	modelName := model.GetName()
	switch modelName {
	case "index":
		return &chatTranslator{model}, nil
	case "qwen":
		return &chatTranslator{model}, nil
	default:
		return nil, ErrInvalidModelForTranslation
	}
}

var sysPromptIndexTranslation = "翻译%s到%s，不要换行，回复请以\"翻译结果：\"开头。"

var translationExamples = []map[Lang]string{
	{
		English:  "Today is Sunday.",
		Chinese:  "今天是星期日。",
		Japanese: "今日は日曜日です。",
	},
	{
		English:  "Hello.",
		Chinese:  "你好。",
		Japanese: "ごきげんよう。",
	},
	{
		English:  "Hello, World.",
		Chinese:  "你好，世界。",
		Japanese: "こんにちは世界。",
	},
}

func (l *chatTranslator) IsSupport(source, target Lang) bool {
	var sourceOk, targetOk bool

	switch source {
	case English:
		fallthrough
	case Japanese:
		sourceOk = true
	default:
		sourceOk = false
	}

	switch target {
	case Chinese:
		sourceOk = true
	default:
		sourceOk = false
	}

	return sourceOk && targetOk
}

func (l *chatTranslator) getLangString(lang Lang) (name string, err error) {
	switch lang {
	case Chinese:
		return "中文", nil
	case English:
		return "英文", nil
	case Japanese:
		return "日语", nil
	default:
		return "", ErrUnexpectedLanguage
	}
}

func (l *chatTranslator) getSysPrompt(source, target Lang) (prompt string, err error) {
	srcLang, err := l.getLangString(source)
	if err != nil {
		return
	}

	tgtLang, err := l.getLangString(target)
	if err != nil {
		return
	}

	return fmt.Sprintf(sysPromptIndexTranslation, srcLang, tgtLang), nil
}

func (l *chatTranslator) Translate(request TranslationRequest) (llm.Completion, error) {
	chat, err := NewChat(l.model)
	if err != nil {
		return nil, err
	}

	sysPrompt, err := l.getSysPrompt(request.SourceLang, request.TargetLang)
	if err != nil {
		return nil, err
	}

	exampleIdx := rand.Intn(len(translationExamples))
	leftCtxSrc := translationExamples[exampleIdx][request.SourceLang]
	leftCtxTgt := translationExamples[exampleIdx][request.TargetLang]
	if request.LeftContextSource != "" {
		leftCtxSrc = request.LeftContextSource
		leftCtxTgt = request.LeftContextTarget
	}

	messages := []Message{
		{"system", sysPrompt},
		{"user", leftCtxSrc + request.Text},
		{"assistant", "翻译结果：" + leftCtxTgt},
	}

	if request.Temperature > 0 {
		chat.SetTemperature(request.Temperature)
	}

	return chat.Chat(messages)
}
