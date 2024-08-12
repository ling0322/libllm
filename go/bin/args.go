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
	"log"
	"log/slog"
	"os"
	"strings"

	"github.com/ling0322/libllm/go/llm"
	"github.com/ling0322/libllm/go/skill"
)

type binArgs struct {
	fs *flag.FlagSet

	modelPath  string
	device     string
	inputFile  string
	outputFile string
	lang       string
	targetLang string
}

func newBinArgs(fs *flag.FlagSet) *binArgs {
	return &binArgs{
		fs: fs,
	}
}

func (a *binArgs) addDeviceFlag() {
	a.fs.StringVar(&a.device, "device", "auto", "inference device, either cpu, cuda or auto")
}

func (a *binArgs) getDevice() llm.Device {
	var device llm.Device
	if strings.ToLower(a.device) == "cpu" {
		device = llm.Cpu
	} else if strings.ToLower(a.device) == "cuda" {
		device = llm.Cuda
	} else if strings.ToLower(a.device) == "auto" {
		device = llm.Auto
	} else {
		log.Fatalf("unexpected device %s", a.device)
	}

	return device
}

func (a *binArgs) addModelFlag() {
	a.fs.StringVar(&a.modelPath, "m", "", "the libllm model, it could be model name or model file,"+
		" model files are with suffix \".llmpkg\". "+
		"\nFor some specific tasks like transcription-translation, it could be a list of"+
		" models, seperated by \",\".")
}

func (a *binArgs) getModel() string {
	if a.modelPath == "" {
		slog.Error("model name (-m) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	models := a.splitModels()
	if len(models) != 1 {
		slog.Error("only 1 model (-m) is expected, please check if there is any unexpected comma" +
			" \",\" in model arg (-m).")
		a.fs.Usage()
		os.Exit(1)
	}

	return models[0]
}

func (a *binArgs) splitModels() []string {
	models := strings.Split(a.modelPath, ",")
	for _, model := range models {
		if model == "" {
			slog.Error("invalid model name (-m).")
			a.fs.Usage()
			os.Exit(1)
		}
	}

	return models
}

func (a *binArgs) getNumModels() int {
	return len(a.splitModels())
}

func (a *binArgs) getModelList() []string {
	if a.modelPath == "" {
		slog.Error("model name (-m) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	return a.splitModels()
}

func (a *binArgs) addInputFlag() {
	a.fs.StringVar(&a.inputFile, "i", "", "the input file.")
}

func (a *binArgs) getInput() string {
	if a.inputFile == "" {
		slog.Error("input file (-i) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	return a.inputFile
}

func (a *binArgs) addOutputFlag() {
	a.fs.StringVar(&a.outputFile, "o", "", "the output file.")
}

func (a *binArgs) getOutput() string {
	if a.outputFile == "" {
		slog.Error("output file (-o) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	return a.outputFile
}

func (a *binArgs) addLangFlag() {
	a.fs.StringVar(&a.lang, "lang", "", "the language of input.")
}

func (a *binArgs) getLang() skill.Lang {
	if a.lang == "" {
		slog.Error("language (-lang) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	lang, err := skill.ParseLang(a.lang)
	if err != nil {
		log.Fatal(err)
	}

	return lang
}

func (a *binArgs) addTargetLangFlag() {
	a.fs.StringVar(&a.targetLang, "targetlang", "", "the target language.")
}

func (a *binArgs) getTargetLang() skill.Lang {
	if a.targetLang == "" {
		slog.Error("target language (-targetlang) is empty.")
		a.fs.Usage()
		os.Exit(1)
	}

	lang, err := skill.ParseLang(a.targetLang)
	if err != nil {
		log.Fatal(err)
	}

	return lang
}
