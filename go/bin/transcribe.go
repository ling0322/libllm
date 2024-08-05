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
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/asticode/go-astisub"
	"github.com/ling0322/libllm/go/llm"
	"github.com/ling0322/libllm/go/skill"
)

func printTranscribeUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, "Usage: llm transcribe [OPTIONS]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Options:")
	fs.PrintDefaults()
	fmt.Fprintln(os.Stderr, "")
}

func TranscriptionToSubtitle(transcriptions []skill.TranscriptionResult) *astisub.Subtitles {
	s := astisub.NewSubtitles()
	for index, t := range transcriptions {
		s.Items = append(s.Items, &astisub.Item{
			Index:   index,
			EndAt:   t.End,
			Lines:   []astisub.Line{{Items: []astisub.LineItem{{Text: t.Text}}}},
			StartAt: t.Begin,
		})
	}

	return s
}

func translateTranscription(
	translationModel string,
	device llm.Device,
	srcLang, tgtLang skill.Lang,
	transcriptions []skill.TranscriptionResult) ([]skill.TranscriptionResult, error) {

	model, err := createModelAutoDownload(translationModel, device)
	if err != nil {
		log.Fatal(err)
	}

	translator, err := skill.NewTranslator(model)
	if err != nil {
		log.Fatal(err)
	}

	merger := skill.NewTranscriptionMerger(skill.TranscriptionMergeOptions{
		Language: skill.English,
	})
	transcriptions = merger.Merge(transcriptions)
	translatedTrxn := []skill.TranscriptionResult{}
	for _, t := range transcriptions {
		tr, err := translate(translator, srcLang, tgtLang, t.Text, nil)
		if err != nil {
			return nil, err
		}
		slog.Info(fmt.Sprintf("translate \"%s\" to \"%s\"", t.Text, tr.tgtText))

		t.Text = tr.tgtText
		translatedTrxn = append(translatedTrxn, t)
	}

	return translatedTrxn, nil
}

func transcribeMain(args []string) {
	fs := flag.NewFlagSet("", flag.ExitOnError)
	fs.Usage = func() {
		printTranscribeUsage(fs)
	}

	ba := newBinArgs(fs)
	ba.addDeviceFlag()
	ba.addModelFlag()
	ba.addInputFlag()
	ba.addOutputFlag()
	ba.addLangFlag()
	ba.addTargetLangFlag()
	_ = fs.Parse(args)

	var modelFile, translationModelFile string
	var srcLang, tgtLang skill.Lang
	if ba.getNumModels() == 2 {
		models := ba.getModelList()
		modelFile = models[0]
		translationModelFile = models[1]

		srcLang = ba.getLang()
		tgtLang = ba.getTargetLang()
	} else if ba.getNumModels() == 1 {
		modelFile = ba.getModel()
	} else {
		slog.Error("invalid number of models (-m)")
		fs.Usage()
		os.Exit(1)
	}

	device := ba.getDevice()
	inputFile := ba.getInput()
	outputFile := ba.getOutput()

	if fs.NArg() != 0 {
		fs.Usage()
		os.Exit(1)
	}

	model, err := createModelAutoDownload(modelFile, device)
	if err != nil {
		log.Fatal(err)
	}

	fd, err := os.Open(inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer fd.Close()

	d0 := time.Now()
	transcriber := skill.NewWhisperTranscriber(model, fd)
	transcriptions := []skill.TranscriptionResult{}
	for transcriber.Transcribe() {
		r := transcriber.Result()
		transcriptions = append(transcriptions, r)
		slog.Info(r.String())
	}

	if err = transcriber.Err(); err != nil {
		log.Fatal(err)
	}

	processingTime := time.Since(d0)
	slog.Info(
		fmt.Sprintf("processed %s audio in %s, rtf=%.3f",
			transcriber.Offset(),
			processingTime.Round(time.Millisecond),
			processingTime.Seconds()/transcriber.Offset().Seconds()))

	transcriber.Dispose()
	model.Dispose()

	if translationModelFile != "" {
		transcriptions, err = translateTranscription(
			translationModelFile, device, srcLang, tgtLang, transcriptions)
	}
	if err != nil {
		log.Fatal(err)
	}

	slog.Info(fmt.Sprintf("save file to %s", outputFile))
	subtitle := TranscriptionToSubtitle(transcriptions)
	err = subtitle.Write(outputFile)
	if err != nil {
		log.Fatal(err)
	}

}
