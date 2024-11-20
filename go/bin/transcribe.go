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
	"path/filepath"
	"strings"

	"github.com/asticode/go-astisub"
	"github.com/ling0322/libllm/go/llm"
	"github.com/ling0322/libllm/go/skill"
)

type translationConfig struct {
	srcLang   skill.Lang
	tgtLang   skill.Lang
	modelName string
	device    string
}

func printTranscribeUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, "Usage: llm transcribe [OPTIONS]")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Options:")
	fs.PrintDefaults()
	fmt.Fprintln(os.Stderr, "")
}

func TranscriptionToSubtitle(results []llm.RecognitionResult) *astisub.Subtitles {
	s := astisub.NewSubtitles()
	for index, t := range results {
		s.Items = append(s.Items, &astisub.Item{
			Index:   index,
			EndAt:   t.End,
			Lines:   []astisub.Line{{Items: []astisub.LineItem{{Text: t.Text}}}},
			StartAt: t.Begin,
		})
	}

	return s
}

func getTranscriptionLang(results []llm.RecognitionResult) skill.Lang {
	if len(results) == 0 {
		return skill.UnknownLanguage
	}

	langCount := map[string]int{}
	for _, tx := range results {
		langCount[tx.Language] += 1
	}

	maxLang := ""
	maxCount := 0
	for lang, count := range langCount {
		if count >= maxCount {
			maxLang = lang
			maxCount = count
		}
	}

	lang, err := skill.ParseLang(maxLang)
	slog.Info(fmt.Sprintf("transcription language is %s", lang.String()))
	if err != nil {
		return skill.UnknownLanguage
	} else {
		return lang
	}
}

func saveTranscription(transcriptions []llm.RecognitionResult, filename string) error {
	slog.Info(fmt.Sprintf("save transcription to %s", filename))
	subtitle := TranscriptionToSubtitle(transcriptions)
	err := subtitle.Write(filename)
	if err != nil {
		return err
	}

	return nil
}

func getOutputFile(ba *binArgs) string {
	outputFile := ba.tryGetOutput()
	if outputFile != "" {
		return outputFile
	}

	// otherwise using <input-filename>.srt
	inputFile := ba.getInput()
	basename := strings.TrimSuffix(filepath.Base(inputFile), filepath.Ext(inputFile))
	dirname := filepath.Dir(inputFile)
	outputFile = filepath.Join(dirname, basename+".srt")

	return outputFile
}

func getTranscriotionModel(ba *binArgs) string {
	if ba.getNumModels() == 2 {
		models := ba.getModels()
		return models[0]
	} else if ba.getNumModels() == 1 {
		return ba.getModel()
	} else if ba.getNumModels() == 0 {
		return "whisper"
	} else {
		slog.Error("invalid number of models (-m)")
		os.Exit(1)
	}

	return ""
}

func getTranslationModel(ba *binArgs) string {
	if ba.getNumModels() == 2 {
		models := ba.getModels()
		return models[1]
	} else if ba.getNumModels() == 1 || ba.getNumModels() == 0 {
		return "index"
	} else {
		slog.Error("invalid number of models (-m)")
		os.Exit(1)
	}

	return ""
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

	var tgtLang skill.Lang
	modelFile := getTranscriotionModel(ba)
	tgtLang = ba.getTargetLang()
	device := ba.getDevice()
	inputFile := ba.getInput()
	outputFile := getOutputFile(ba)

	if fs.NArg() != 0 {
		fs.Usage()
		os.Exit(1)
	}

	model, err := createASRModelAutoDownload(modelFile, device)
	if err != nil {
		log.Fatal(err)
	}

	slog.Info(fmt.Sprintf("output file is %s", outputFile))

	recognition, err := model.Recognize(inputFile)
	if err != nil {
		log.Fatal(err)
	}

	transcriptions := []llm.RecognitionResult{}
	for recognition.Next() {
		r := recognition.Result()
		transcriptions = append(transcriptions, r)
		slog.Info(r.String())
	}

	if err = recognition.Err(); err != nil {
		log.Fatal(err)
	}

	recognition.Dispose()
	model.Dispose()

	srcLang := getTranscriptionLang(transcriptions)
	if srcLang != skill.UnknownLanguage && tgtLang != skill.UnknownLanguage && srcLang != tgtLang {
		// save transcription (without translation)
		fileExt := filepath.Ext(outputFile)
		fileBaseName := outputFile[:len(outputFile)-len(fileExt)]
		outputTxFile := fmt.Sprintf("%s.%s%s", fileBaseName, srcLang.String(), fileExt)
		err = saveTranscription(transcriptions, outputTxFile)
		if err != nil {
			log.Fatal(err)
		}

		// translation
		config := translationConfig{
			srcLang,
			tgtLang,
			getTranslationModel(ba),
			ba.getDevice(),
		}
		err := TranslateSubtitle(config, outputTxFile, outputFile)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		err = saveTranscription(transcriptions, outputFile)
		if err != nil {
			log.Fatal(err)
		}
	}
}
