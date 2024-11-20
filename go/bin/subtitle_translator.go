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

package main

import (
	"fmt"
	"log"
	"log/slog"
	"strings"

	"github.com/asticode/go-astisub"
	"github.com/ling0322/libllm/go/llm"
	"github.com/ling0322/libllm/go/skill"
)

type transcriptionTranslator struct {
	historySrc []string
	historyTgt []string
	translator *skill.Translator
	srcLang    skill.Lang
	tgtLang    skill.Lang
}

func (t *transcriptionTranslator) translateOneInternal(
	text string, temperature float32) (translationResult, error) {

	req := skill.TranslationRequest{
		Text:              text,
		LeftContextSource: strings.Join(t.historySrc, " "),
		LeftContextTarget: strings.Join(t.historyTgt, " "),
		SourceLang:        t.srcLang,
		TargetLang:        t.tgtLang,
		Temperature:       temperature,
	}
	tr, err := translate(t.translator, req, nil)
	if err != nil {
		return translationResult{}, err
	}
	return tr, nil
}

func (t *transcriptionTranslator) translateOneWithRetry(text string) (translationResult, error) {
	var tr translationResult
	var err error
	var temperature float32 = 1.0
	for n := 0; n < 5; n++ {
		tr, err = t.translateOneInternal(text, temperature)
		if err != nil {
			return translationResult{}, err
		}

		tr.tgtText = strings.ReplaceAll(tr.tgtText, "\n", " ")
		tr.tgtText = strings.TrimSpace(tr.tgtText)
		if tr.tgtText != "" {
			return tr, nil
		}

		// clear history if no result.
		t.historySrc = []string{}
		t.historyTgt = []string{}
	}

	return tr, nil
}

func (t *transcriptionTranslator) translateOne(text string) (translationResult, error) {
	tr, err := t.translateOneWithRetry(text)
	if err != nil {
		return translationResult{}, err
	}

	if tr.tgtText == "" {
		t.historySrc = []string{}
		t.historyTgt = []string{}
		tr.tgtText = tr.srcText
		return tr, nil
	}

	t.historySrc = append(t.historySrc, text)
	t.historyTgt = append(t.historyTgt, tr.tgtText)
	if len(t.historySrc) > 3 {
		t.historySrc = t.historySrc[1:]
		t.historyTgt = t.historyTgt[1:]
	}

	return tr, nil
}

func (t *transcriptionTranslator) translate(
	transcriptions []llm.RecognitionResult) ([]llm.RecognitionResult, error) {
	translatedTrxn := []llm.RecognitionResult{}
	for _, transcription := range transcriptions {
		tr, err := t.translateOne(transcription.Text)
		if err != nil {
			return nil, err
		}

		translated := tr.tgtText
		translated = strings.TrimSpace(translated)
		translated = strings.Replace(translated, "\n", " ", -1)
		slog.Info(fmt.Sprintf("translate \"%s\" to \"%s\"", tr.srcText, translated))

		transcription.Text = translated
		translatedTrxn = append(translatedTrxn, transcription)
	}

	return translatedTrxn, nil
}

func newTranscriptionTranslator(
	translationModel string,
	device string,
	srcLang, tgtLang skill.Lang) (*transcriptionTranslator, error) {
	model, err := createModelAutoDownload(translationModel, device)
	if err != nil {
		return nil, err
	}

	translator, err := skill.NewTranslator(model)
	if err != nil {
		return nil, err
	}

	return &transcriptionTranslator{
		translator: translator,
		srcLang:    srcLang,
		tgtLang:    tgtLang,
	}, nil
}

// read a subtitle file to
func ReadSubtitleFile(filename string) ([]llm.RecognitionResult, error) {
	subtitles, err := astisub.OpenFile(filename)
	if err != nil {
		return nil, err
	}

	txs := []llm.RecognitionResult{}
	for _, item := range subtitles.Items {
		lines := []string{}
		for _, line := range item.Lines {
			lines = append(lines, line.String())
		}
		tx := llm.RecognitionResult{
			Begin: item.StartAt,
			End:   item.EndAt,
			Text:  strings.Join(lines, " "),
		}

		txs = append(txs, tx)
	}

	return txs, nil
}

func TranslateSubtitle(config translationConfig, inputFile, outputFile string) error {
	tt, err := newTranscriptionTranslator(
		config.modelName,
		config.device,
		config.srcLang,
		config.tgtLang)
	if err != nil {
		return err
	}

	txs, err := ReadSubtitleFile(inputFile)
	if err != nil {
		return err
	}

	txs, err = tt.translate(txs)
	if err != nil {
		return err
	}

	err = saveTranscription(txs, outputFile)
	if err != nil {
		log.Fatal(err)
	}

	return nil
}
